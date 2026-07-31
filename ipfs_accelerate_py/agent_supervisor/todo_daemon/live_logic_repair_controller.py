"""Edge orchestration for live logic-repair (LPR-017).

The controller lives in ``todo_daemon/`` so it can inject pure analysis, proof,
planning, and validation callbacks without reversing the package DAG.  It is
versioned and feature-gated (default off): legacy artifact-supplied RPR and
ordinary proposal flows remain unchanged when the flag is disabled.

Hard invariants:

* feature flag defaults off;
* analytical success never invokes a model/provider;
* model calls use only LPR-016 context overlays projected into existing RPR
  packets (never a third write authority);
* every ordinary provider proposal is analyzed as a read-only candidate
  overlay before mutation;
* signature changes that omit resolved callers are rejected, deferred for
  expansion/re-admission, or expanded into a newly admitted write set;
* required unknown frontier always abstains;
* pre-provider/proposal gates revalidate roots, receipts, scope, and lease;
* no direct write bypass exists;
* optional heavy modules are imported lazily (cold import path).
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final


LIVE_LOGIC_REPAIR_CONTROLLER_INTERFACE: Final[str] = "LiveLogicRepairController@1"
LIVE_LOGIC_REPAIR_CONTROLLER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/live-logic-repair-controller@1"
)
LIVE_LOGIC_REPAIR_CONTROLLER_VERSION: Final[int] = 1
PRODUCER_ID: Final[str] = "live-logic-repair-controller@1"

# Contract-repair explicit path (broken trace → target admission).
CONTRACT_REPAIR_STAGE_ORDER: Final[tuple[str, ...]] = (
    "trace",
    "contracts",
    "retrieval",
    "goal",
    "corpus",
    "tactician",
    "hypothesis",
    "gate",
    "lowering",
    "hammer",
    "refinement",
    "admission",
    "target_admission",
)

# Intentional change / propagation path (delta → atomic plan admission).
CHANGE_PROPAGATION_STAGE_ORDER: Final[tuple[str, ...]] = (
    "delta",
    "graph",
    "impact",
    "consumer",
    "value",
    "behavior",
    "goal",
    "corpus",
    "tactician",
    "hypothesis",
    "gate",
    "lowering",
    "hammer",
    "refinement",
    "admission",
    "atomic_plan_admission",
)

# Ordinary provider-proposal overlay intake (before patch application).
PROPOSAL_OVERLAY_STAGE_ORDER: Final[tuple[str, ...]] = (
    "overlay_materialize",
    "contract_delta",
    "impact_closure",
    "consumer_frontier",
    "caller_disposition",
    "admit_or_reject",
)


class LiveLogicRepairError(ValueError):
    """A typed fail-closed live logic-repair orchestration error."""


class LiveLogicRepairMode(str, Enum):
    """Which primary analysis spine the controller runs."""

    CONTRACT_REPAIR = "contract_repair"
    CHANGE_PROPAGATION = "change_propagation"
    PROPOSAL_OVERLAY = "proposal_overlay"


class LiveLogicRepairDisposition(str, Enum):
    """Closed outcomes for one controller run."""

    DISABLED = "disabled"
    ADMITTED = "admitted"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    ABSTAINED = "abstained"
    EXPANDED = "expanded"
    MALFORMED = "malformed"


class OverlayCallerDisposition(str, Enum):
    """Per-caller disposition before mutation is allowed."""

    IN_WRITE_SET = "in_write_set"
    COMPATIBILITY_PROOF = "compatibility_proof"
    NO_CHANGE_PROOF = "no_change_proof"
    OMITTED = "omitted"
    UNKNOWN = "unknown"
    EXPANDED = "expanded"


class OverlayGateDisposition(str, Enum):
    """Outcome of the candidate-overlay contract-delta gate."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    EXPANDED = "expanded"
    ABSTAINED = "abstained"
    DISABLED = "disabled"


@dataclass(frozen=True)
class LiveLogicRepairPolicy:
    """Feature gate and optional-provider controls for live logic repair."""

    enable_live_logic_repair: bool = False
    allow_provider_for_model_steps: bool = True
    analytical_skips_provider: bool = True
    reject_omitted_callers: bool = True
    expand_write_set_on_omission: bool = True
    require_unknown_frontier_abstain: bool = True
    revalidate_roots_and_receipts: bool = True

    def __post_init__(self) -> None:
        for name in (
            "enable_live_logic_repair",
            "allow_provider_for_model_steps",
            "analytical_skips_provider",
            "reject_omitted_callers",
            "expand_write_set_on_omission",
            "require_unknown_frontier_abstain",
            "revalidate_roots_and_receipts",
        ):
            if not isinstance(getattr(self, name), bool):
                raise LiveLogicRepairError(f"{name} must be a boolean")

    @classmethod
    def from_value(
        cls,
        value: "LiveLogicRepairPolicy | Mapping[str, Any] | None",
    ) -> "LiveLogicRepairPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("live logic repair policy must be a mapping")
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                "unknown live logic repair policy fields: " + ", ".join(unknown)
            )
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_live_logic_repair": self.enable_live_logic_repair,
            "allow_provider_for_model_steps": self.allow_provider_for_model_steps,
            "analytical_skips_provider": self.analytical_skips_provider,
            "reject_omitted_callers": self.reject_omitted_callers,
            "expand_write_set_on_omission": self.expand_write_set_on_omission,
            "require_unknown_frontier_abstain": (
                self.require_unknown_frontier_abstain
            ),
            "revalidate_roots_and_receipts": self.revalidate_roots_and_receipts,
        }


@dataclass(frozen=True)
class CallableSignatureDelta:
    """Base-to-proposal callable signature change (AST-derived)."""

    symbol: str
    path: str
    before_params: tuple[str, ...]
    after_params: tuple[str, ...]
    before_signature: str
    after_signature: str

    @property
    def arity_increased(self) -> bool:
        return len(self.after_params) > len(self.before_params)

    @property
    def changed(self) -> bool:
        return self.before_signature != self.after_signature

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "path": self.path,
            "before_params": list(self.before_params),
            "after_params": list(self.after_params),
            "before_signature": self.before_signature,
            "after_signature": self.after_signature,
            "arity_increased": self.arity_increased,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class CallerDispositionRecord:
    """Disposition of one resolved caller before mutation."""

    caller_id: str
    path: str
    symbol: str
    disposition: OverlayCallerDisposition
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            OverlayCallerDisposition(self.disposition),
        )
        for name in ("caller_id", "path", "symbol"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise LiveLogicRepairError(f"{name} is required")
            object.__setattr__(self, name, value.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "caller_id": self.caller_id,
            "path": self.path,
            "symbol": self.symbol,
            "disposition": self.disposition.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CandidateOverlayReceipt:
    """Read-only candidate overlay analysis (no write authority)."""

    overlay_id: str
    proposal_id: str
    repository_id: str
    base_tree_id: str
    candidate_tree_id: str
    changed_paths: tuple[str, ...]
    write_set: tuple[str, ...]
    signature_deltas: tuple[CallableSignatureDelta, ...]
    resolved_callers: tuple[str, ...]
    omitted_callers: tuple[str, ...]
    unknown_frontier: tuple[str, ...]
    caller_dispositions: tuple[CallerDispositionRecord, ...]
    expanded_write_set: tuple[str, ...] = ()
    impact_closure_id: str = ""
    consumer_frontier_id: str = ""
    delta_id: str = ""
    read_only: bool = True
    mutation_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.read_only:
            raise LiveLogicRepairError(
                "candidate overlay must remain read-only; no write authority"
            )
        for name in (
            "overlay_id",
            "proposal_id",
            "repository_id",
            "base_tree_id",
            "candidate_tree_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise LiveLogicRepairError(f"{name} is required")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "changed_paths", tuple(self.changed_paths))
        object.__setattr__(self, "write_set", tuple(self.write_set))
        object.__setattr__(
            self, "signature_deltas", tuple(self.signature_deltas)
        )
        object.__setattr__(
            self, "resolved_callers", tuple(self.resolved_callers)
        )
        object.__setattr__(self, "omitted_callers", tuple(self.omitted_callers))
        object.__setattr__(
            self, "unknown_frontier", tuple(self.unknown_frontier)
        )
        object.__setattr__(
            self, "caller_dispositions", tuple(self.caller_dispositions)
        )
        object.__setattr__(
            self, "expanded_write_set", tuple(self.expanded_write_set)
        )
        if not isinstance(self.mutation_allowed, bool):
            raise LiveLogicRepairError("mutation_allowed must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "candidate-overlay-receipt@1"
            ),
            "overlay_id": self.overlay_id,
            "proposal_id": self.proposal_id,
            "repository_id": self.repository_id,
            "base_tree_id": self.base_tree_id,
            "candidate_tree_id": self.candidate_tree_id,
            "changed_paths": list(self.changed_paths),
            "write_set": list(self.write_set),
            "signature_deltas": [d.to_dict() for d in self.signature_deltas],
            "resolved_callers": list(self.resolved_callers),
            "omitted_callers": list(self.omitted_callers),
            "unknown_frontier": list(self.unknown_frontier),
            "caller_dispositions": [
                d.to_dict() for d in self.caller_dispositions
            ],
            "expanded_write_set": list(self.expanded_write_set),
            "impact_closure_id": self.impact_closure_id,
            "consumer_frontier_id": self.consumer_frontier_id,
            "delta_id": self.delta_id,
            "read_only": self.read_only,
            "mutation_allowed": self.mutation_allowed,
        }


@dataclass(frozen=True)
class CandidateOverlayGateResult:
    """Gate result for one ordinary provider proposal overlay."""

    disposition: OverlayGateDisposition
    detail: str = ""
    reason_codes: tuple[str, ...] = ()
    stages_completed: tuple[str, ...] = ()
    overlay: CandidateOverlayReceipt | None = None
    provider_invoked: bool = False
    mutation_allowed: bool = False
    expanded_write_set: tuple[str, ...] = ()
    proof_bundle: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "disposition", OverlayGateDisposition(self.disposition)
        )
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(
            self, "stages_completed", tuple(self.stages_completed)
        )
        object.__setattr__(
            self, "expanded_write_set", tuple(self.expanded_write_set)
        )
        if self.provider_invoked:
            raise LiveLogicRepairError(
                "candidate overlay gate must never invoke a provider"
            )

    @property
    def admitted(self) -> bool:
        return self.disposition in {
            OverlayGateDisposition.ADMITTED,
            OverlayGateDisposition.EXPANDED,
        } and self.mutation_allowed

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "detail": self.detail,
            "reason_codes": list(self.reason_codes),
            "stages_completed": list(self.stages_completed),
            "overlay": self.overlay.to_dict() if self.overlay else None,
            "provider_invoked": self.provider_invoked,
            "mutation_allowed": self.mutation_allowed,
            "expanded_write_set": list(self.expanded_write_set),
            "proof_bundle_id": (
                str(
                    getattr(self.proof_bundle, "candidate_id", "")
                    or getattr(self.proof_bundle, "bundle_id", "")
                    or ""
                )
            ),
        }


@dataclass(frozen=True)
class LiveLogicRepairRequest:
    """Bound stage artifacts for one feature-gated live repair run.

    Stages are supplied as already-bound local artifacts so hermetic tests can
    enforce order without granting optional providers any role in admission.
    Live extractors may be supplied as pure callbacks without changing the
    orchestration contract.
    """

    mode: LiveLogicRepairMode | str
    repository_id: str
    tree_id: str
    # Contract-repair spine (optional when mode is change/overlay).
    trace: Any = None
    contracts: Any = None
    candidates: Sequence[Any] = ()
    # Change-propagation spine.
    delta: Any = None
    graph_id: str = ""
    impact_closure: Any = None
    consumers: Sequence[Any] = ()
    value_proofs: Sequence[Any] = ()
    behavior_gaps: Sequence[Any] = ()
    # Shared logic stages (goal → admission).
    goals: Sequence[Any] = ()
    corpus: Any = None
    tactician_plan: Any = None
    hypotheses: Sequence[Any] = ()
    plan_gate_receipt: Any = None
    lowering: Any = None
    hammer_receipt: Any = None
    refinement: Any = None
    prediction_decision: Any = None
    prediction_receipts: Sequence[Any] = ()
    # Existing RPR admission inputs.
    target_admission: Any = None
    atomic_plan_admission: Any = None
    evidence_bundle: Any = None
    # Proposal overlay inputs.
    proposal: Any = None
    proposal_id: str = ""
    write_set: Sequence[str] = ()
    base_sources: Mapping[str, str] = field(default_factory=dict)
    candidate_sources: Mapping[str, str] = field(default_factory=dict)
    resolved_callers: Sequence[Mapping[str, Any] | str] = ()
    unknown_frontier: Sequence[str] = ()
    compatibility_proofs: Sequence[str] = ()
    no_change_proofs: Sequence[str] = ()
    # Roots / receipts for revalidation.
    roots: Any = None
    logic_roots: Any = None
    writer_lease: Any = None
    scope_paths: Sequence[str] = ()
    # Optional base proof bundle to compose with predictions.
    base_proof_bundle: Any = None
    # Optional pure callbacks (never used for writes).
    stage_callbacks: Mapping[str, Callable[..., Any]] = field(
        default_factory=dict
    )
    # When true, attempt model path via LPR-016 overlay only.
    model_required: bool = False
    analytical_success: bool = True
    task_id: str = ""
    now: int = 0

    def __post_init__(self) -> None:
        mode = self.mode
        if isinstance(mode, str):
            mode = LiveLogicRepairMode(mode)
        object.__setattr__(self, "mode", mode)
        for name in ("repository_id", "tree_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise LiveLogicRepairError(f"{name} is required")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "consumers", tuple(self.consumers))
        object.__setattr__(self, "value_proofs", tuple(self.value_proofs))
        object.__setattr__(self, "behavior_gaps", tuple(self.behavior_gaps))
        object.__setattr__(self, "goals", tuple(self.goals))
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))
        object.__setattr__(
            self, "prediction_receipts", tuple(self.prediction_receipts)
        )
        object.__setattr__(self, "write_set", tuple(self.write_set))
        object.__setattr__(
            self, "resolved_callers", tuple(self.resolved_callers)
        )
        object.__setattr__(
            self, "unknown_frontier", tuple(self.unknown_frontier)
        )
        object.__setattr__(
            self, "compatibility_proofs", tuple(self.compatibility_proofs)
        )
        object.__setattr__(
            self, "no_change_proofs", tuple(self.no_change_proofs)
        )
        object.__setattr__(self, "scope_paths", tuple(self.scope_paths))
        if not isinstance(self.base_sources, Mapping):
            raise LiveLogicRepairError("base_sources must be a mapping")
        if not isinstance(self.candidate_sources, Mapping):
            raise LiveLogicRepairError("candidate_sources must be a mapping")
        object.__setattr__(self, "base_sources", dict(self.base_sources))
        object.__setattr__(
            self, "candidate_sources", dict(self.candidate_sources)
        )
        if not isinstance(self.stage_callbacks, Mapping):
            raise LiveLogicRepairError("stage_callbacks must be a mapping")
        object.__setattr__(self, "stage_callbacks", dict(self.stage_callbacks))
        for name in ("model_required", "analytical_success"):
            if not isinstance(getattr(self, name), bool):
                raise LiveLogicRepairError(f"{name} must be a boolean")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LiveLogicRepairRequest":
        if not isinstance(value, Mapping):
            raise TypeError("live logic repair request must be a mapping")
        known = set(cls.__dataclass_fields__)
        payload = {k: v for k, v in value.items() if k in known}
        return cls(**payload)


@dataclass(frozen=True)
class LiveLogicRepairResult:
    """Outcome of one feature-gated live logic-repair controller run."""

    enabled: bool
    mode: str
    stage: str
    disposition: str
    detail: str = ""
    provider_invoked: bool = False
    stages_completed: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    proof_bundle: Any = None
    prediction_decision: Any = None
    overlay_gate: CandidateOverlayGateResult | None = None
    packet: Any = None
    write_paths: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    target_admission: Any = None
    atomic_plan_admission: Any = None
    model_context_overlay: Any = None
    mutation_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stages_completed", tuple(self.stages_completed)
        )
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(self, "write_paths", tuple(self.write_paths))
        object.__setattr__(self, "read_paths", tuple(self.read_paths))

    @property
    def admitted(self) -> bool:
        return self.disposition in {
            LiveLogicRepairDisposition.ADMITTED.value,
            LiveLogicRepairDisposition.EXPANDED.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LIVE_LOGIC_REPAIR_CONTROLLER_SCHEMA,
            "interface": LIVE_LOGIC_REPAIR_CONTROLLER_INTERFACE,
            "version": LIVE_LOGIC_REPAIR_CONTROLLER_VERSION,
            "enabled": self.enabled,
            "mode": self.mode,
            "stage": self.stage,
            "disposition": self.disposition,
            "detail": self.detail,
            "provider_invoked": self.provider_invoked,
            "stages_completed": list(self.stages_completed),
            "reason_codes": list(self.reason_codes),
            "write_paths": list(self.write_paths),
            "read_paths": list(self.read_paths),
            "mutation_allowed": self.mutation_allowed,
            "overlay_gate": (
                self.overlay_gate.to_dict() if self.overlay_gate else None
            ),
        }


def _disabled_result(
    mode: str = LiveLogicRepairMode.CONTRACT_REPAIR.value,
    detail: str = "enable_live_logic_repair is false",
) -> LiveLogicRepairResult:
    return LiveLogicRepairResult(
        enabled=False,
        mode=mode,
        stage="disabled",
        disposition=LiveLogicRepairDisposition.DISABLED.value,
        detail=detail,
        provider_invoked=False,
        mutation_allowed=False,
    )


def _fail(
    *,
    mode: str,
    stage: str,
    disposition: str,
    detail: str,
    completed: Sequence[str] = (),
    reason_codes: Sequence[str] = (),
    provider_invoked: bool = False,
    proof_bundle: Any = None,
    overlay_gate: CandidateOverlayGateResult | None = None,
    prediction_decision: Any = None,
    write_paths: Sequence[str] = (),
) -> LiveLogicRepairResult:
    return LiveLogicRepairResult(
        enabled=True,
        mode=mode,
        stage=stage,
        disposition=disposition,
        detail=detail,
        provider_invoked=provider_invoked,
        stages_completed=tuple(completed),
        reason_codes=tuple(reason_codes),
        proof_bundle=proof_bundle,
        prediction_decision=prediction_decision,
        overlay_gate=overlay_gate,
        write_paths=tuple(write_paths),
        mutation_allowed=False,
    )


# ---------------------------------------------------------------------------
# Signature / caller analysis (hermetic AST)
# ---------------------------------------------------------------------------


_DEF_RE = re.compile(
    r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*:",
    re.MULTILINE | re.DOTALL,
)


def _param_names_from_args(args: ast.arguments) -> tuple[str, ...]:
    names: list[str] = []
    for item in list(args.posonlyargs) + list(args.args):
        names.append(item.arg)
    if args.vararg is not None:
        names.append("*" + args.vararg.arg)
    for item in args.kwonlyargs:
        names.append(item.arg)
    if args.kwarg is not None:
        names.append("**" + args.kwarg.arg)
    return tuple(names)


def extract_python_signatures(source: str) -> dict[str, tuple[str, ...]]:
    """Map top-level and nested function names to parameter tuples."""

    if not isinstance(source, str) or not source.strip():
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to line-oriented extraction for partial overlays.
        found: dict[str, tuple[str, ...]] = {}
        for match in _DEF_RE.finditer(source):
            name = match.group(1)
            raw = match.group(2).replace("\n", " ")
            params = tuple(
                p.strip().split(":")[0].split("=")[0].strip()
                for p in raw.split(",")
                if p.strip() and p.strip() != "self"
            )
            found[name] = params
        return found

    found: dict[str, tuple[str, ...]] = {}

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            params = _param_names_from_args(node.args)
            # Drop implicit self/cls for method surface comparison.
            if params and params[0] in {"self", "cls"}:
                params = params[1:]
            found[node.name] = params
            self.generic_visit(node)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            self.visit_FunctionDef(node)  # type: ignore[arg-type]

    Visitor().visit(tree)
    return found


def compute_signature_deltas(
    base_sources: Mapping[str, str],
    candidate_sources: Mapping[str, str],
) -> tuple[CallableSignatureDelta, ...]:
    """Compute base-to-proposal callable signature deltas across paths."""

    deltas: list[CallableSignatureDelta] = []
    paths = sorted(set(base_sources) | set(candidate_sources))
    for path in paths:
        before = extract_python_signatures(base_sources.get(path, ""))
        after = extract_python_signatures(candidate_sources.get(path, ""))
        for symbol in sorted(set(before) | set(after)):
            b_params = before.get(symbol, ())
            a_params = after.get(symbol, ())
            b_sig = f"{symbol}({', '.join(b_params)})"
            a_sig = f"{symbol}({', '.join(a_params)})"
            if b_sig == a_sig:
                continue
            deltas.append(
                CallableSignatureDelta(
                    symbol=symbol,
                    path=path,
                    before_params=b_params,
                    after_params=a_params,
                    before_signature=b_sig,
                    after_signature=a_sig,
                )
            )
    return tuple(deltas)


def find_call_sites(
    sources: Mapping[str, str],
    symbol: str,
) -> tuple[dict[str, str], ...]:
    """Locate call sites of ``symbol`` in source map (path, caller function)."""

    sites: list[dict[str, str]] = []
    for path, source in sources.items():
        if not source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Line-based fallback.
            for match in re.finditer(
                rf"\b{re.escape(symbol)}\s*\(", source
            ):
                sites.append(
                    {
                        "caller_id": f"{path}::callsite:{match.start()}",
                        "path": path,
                        "symbol": symbol,
                    }
                )
            continue

        class CallVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_AsyncFunctionDef(  # noqa: N802
                self, node: ast.AsyncFunctionDef
            ) -> None:
                self.visit_FunctionDef(node)  # type: ignore[arg-type]

            def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name == symbol:
                    caller = self.stack[-1] if self.stack else "<module>"
                    sites.append(
                        {
                            "caller_id": f"{path}::{caller}",
                            "path": path,
                            "symbol": symbol,
                        }
                    )
                self.generic_visit(node)

        CallVisitor().visit(tree)
    return tuple(sites)


def _normalize_caller(
    item: Mapping[str, Any] | str,
    *,
    default_symbol: str = "",
) -> dict[str, str]:
    if isinstance(item, str):
        text = item.strip()
        if "::" in text:
            path, rest = text.split("::", 1)
            return {
                "caller_id": text,
                "path": path,
                "symbol": default_symbol or rest,
            }
        return {
            "caller_id": text,
            "path": text if text.endswith(".py") else "",
            "symbol": default_symbol,
        }
    if not isinstance(item, Mapping):
        raise LiveLogicRepairError("caller entry must be a string or mapping")
    path = str(item.get("path") or "").strip()
    symbol = str(item.get("symbol") or default_symbol or "").strip()
    caller_id = str(item.get("caller_id") or "").strip()
    if not caller_id:
        if path and symbol:
            caller_id = f"{path}::{symbol}"
        else:
            caller_id = path or symbol
    if not caller_id:
        raise LiveLogicRepairError("caller_id is required")
    return {"caller_id": caller_id, "path": path, "symbol": symbol}


def _content_id(payload: Mapping[str, Any]) -> str:
    raw = repr(sorted(payload.items())).encode("utf-8")
    return "cid:" + hashlib.sha256(raw).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Prediction → CandidateProofBundle bridge
# ---------------------------------------------------------------------------


def bridge_predictions_into_proof_bundle(
    *,
    candidate_id: str,
    repository_id: str,
    tree_id: str,
    prediction_decision: Any = None,
    prediction_receipts: Sequence[Any] = (),
    base_proof_bundle: Any = None,
    backend_id: str = "logic-prediction-bridge",
    backend_version: str = "1",
) -> Any:
    """Project admitted logic predictions into a CandidateProofBundle.

    Predictions *compose with* rather than replace an existing proof bundle.
    Projection yields non-conclusive results so proof remains orthogonal and
    inherits the weakest independent source precedence.
    """

    # Lazy import keeps cold daemon paths free of the prover stack.
    from ..proof.contract_repair_prover import (
        CandidateProofBundle,
        CandidateProofResult,
        ContractRepairProofDisposition,
    )
    from ..proof.formal_verification_contracts import (
        AssuranceLevel,
        EvidenceAuthority,
        EvidenceKind,
        EvidenceVerdict,
        ProofEvidence,
        ProofReceipt,
        ProofVerdict,
        ResourceBudget,
    )

    results: list[CandidateProofResult] = []
    if base_proof_bundle is not None:
        base_results = tuple(getattr(base_proof_bundle, "results", ()) or ())
        results.extend(base_results)
        if not candidate_id:
            candidate_id = str(
                getattr(base_proof_bundle, "candidate_id", "") or ""
            )
        if not repository_id:
            repository_id = str(
                getattr(base_proof_bundle, "repository_id", "") or ""
            )
        if not tree_id:
            tree_id = str(getattr(base_proof_bundle, "tree_id", "") or "")
        backend_id = str(
            getattr(base_proof_bundle, "backend_id", "") or backend_id
        )
        backend_version = str(
            getattr(base_proof_bundle, "backend_version", "") or backend_version
        )

    prediction_ids: list[str] = []
    if prediction_decision is not None:
        for attr in (
            "content_id",
            "decision_id",
            "receipt_id",
            "prediction_id",
        ):
            value = getattr(prediction_decision, attr, None)
            if isinstance(value, str) and value.strip():
                prediction_ids.append(value.strip())
                break
        if isinstance(prediction_decision, Mapping):
            for key in (
                "content_id",
                "decision_id",
                "receipt_id",
                "prediction_id",
            ):
                value = prediction_decision.get(key)
                if isinstance(value, str) and value.strip():
                    prediction_ids.append(value.strip())
                    break

    for item in prediction_receipts:
        if isinstance(item, str) and item.strip():
            prediction_ids.append(item.strip())
            continue
        for attr in ("receipt_id", "prediction_id", "content_id"):
            value = getattr(item, attr, None)
            if isinstance(value, str) and value.strip():
                prediction_ids.append(value.strip())
                break
        if isinstance(item, Mapping):
            for key in ("receipt_id", "prediction_id", "content_id"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    prediction_ids.append(value.strip())
                    break

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_ids: list[str] = []
    for pid in prediction_ids:
        if pid not in seen:
            seen.add(pid)
            unique_ids.append(pid)

    if not unique_ids and not results:
        # No predictions and no base: synthesize a single non-conclusive
        # placeholder so the bridge still produces a typed bundle when the
        # controller stage requires one.
        unique_ids = ["prediction:absent"]

    existing_obligation_ids = {
        str(getattr(item, "obligation_id", "") or "") for item in results
    }

    for index, pid in enumerate(unique_ids):
        obligation_id = f"logic-prediction:{pid}"
        if obligation_id in existing_obligation_ids:
            continue
        evidence = ProofEvidence(
            EvidenceKind.STATIC_ANALYSIS,
            EvidenceAuthority.PROVIDER,
            EvidenceVerdict.CANDIDATE,
            artifact_id=f"prediction:{pid}",
            subject_id=obligation_id,
            verifier_id="logic-prediction-bridge@1",
            independent=False,
        )
        receipt = ProofReceipt(
            obligation_id=obligation_id,
            plan_id="plan:logic-prediction-bridge",
            attempt_id=f"attempt:prediction:{index}:{pid}",
            repository_id=repository_id,
            repository_tree_id=tree_id,
            ast_scope_ids=(f"scope:prediction:{pid}",),
            premise_ids=(f"premise:prediction:{pid}",),
            translator_id="translator:logic-prediction-bridge",
            solver_id="logic-prediction",
            kernel_id="kernel:none",
            toolchain_id="toolchain:logic-prediction-bridge",
            policy_id="policy:logic-prediction-bridge",
            resource_budget=ResourceBudget(),
            verdict=ProofVerdict.INCONCLUSIVE,
            evidence=(evidence,),
            provider_claimed_assurance=AssuranceLevel.UNVERIFIED,
        )
        results.append(
            CandidateProofResult(
                obligation_id,
                receipt,
                ContractRepairProofDisposition.NON_CONCLUSIVE,
                ("logic_prediction_projection", "compose_not_replace"),
                f"cache:logic-prediction:{pid}",
            )
        )
        existing_obligation_ids.add(obligation_id)

    if not results:
        raise LiveLogicRepairError(
            "prediction bridge produced no proof results"
        )

    return CandidateProofBundle(
        candidate_id or f"candidate:logic:{tree_id}",
        repository_id,
        tree_id,
        tuple(results),
        backend_id,
        backend_version,
        reason_codes=("logic_prediction_bridge",),
    )


# ---------------------------------------------------------------------------
# Candidate overlay contract-delta gate
# ---------------------------------------------------------------------------


class CandidateOverlayContractDeltaGate:
    """Intercept ordinary provider proposals as read-only candidate overlays.

    Computes base-to-proposal callable contract delta and complete
    impact/consumer frontier.  Signature changes whose write set omits a
    resolved caller are rejected, deferred, or expanded for re-admission.
    Analytical only: never invokes a model.
    """

    INTERFACE: Final[str] = "CandidateOverlayContractDeltaGate@1"

    def __init__(
        self,
        policy: LiveLogicRepairPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        self.policy = LiveLogicRepairPolicy.from_value(policy)

    def evaluate(
        self,
        *,
        proposal_id: str,
        repository_id: str,
        base_tree_id: str,
        candidate_tree_id: str,
        write_set: Sequence[str],
        base_sources: Mapping[str, str],
        candidate_sources: Mapping[str, str],
        resolved_callers: Sequence[Mapping[str, Any] | str] = (),
        unknown_frontier: Sequence[str] = (),
        compatibility_proofs: Sequence[str] = (),
        no_change_proofs: Sequence[str] = (),
        impact_closure_id: str = "",
        consumer_frontier_id: str = "",
        auto_discover_callers: bool = True,
    ) -> CandidateOverlayGateResult:
        if not self.policy.enable_live_logic_repair:
            return CandidateOverlayGateResult(
                disposition=OverlayGateDisposition.DISABLED,
                detail="enable_live_logic_repair is false",
                stages_completed=(),
                mutation_allowed=False,
            )

        completed: list[str] = []
        write_set_t = tuple(
            sorted({str(p).strip() for p in write_set if str(p).strip()})
        )
        if not proposal_id or not repository_id:
            return CandidateOverlayGateResult(
                disposition=OverlayGateDisposition.REJECTED,
                detail="proposal_id and repository_id are required",
                reason_codes=("malformed_overlay_input",),
                mutation_allowed=False,
            )

        # 1. Materialize read-only overlay identity.
        stage = "overlay_materialize"
        overlay_id = _content_id(
            {
                "proposal_id": proposal_id,
                "repository_id": repository_id,
                "base_tree_id": base_tree_id,
                "candidate_tree_id": candidate_tree_id,
                "write_set": list(write_set_t),
            }
        )
        completed.append(stage)

        # 2. Base-to-proposal callable contract delta.
        stage = "contract_delta"
        deltas = compute_signature_deltas(base_sources, candidate_sources)
        delta_id = _content_id(
            {
                "deltas": [d.to_dict() for d in deltas],
                "overlay_id": overlay_id,
            }
        )
        completed.append(stage)

        # 3–4. Impact / consumer frontier (bound or auto-discovered).
        stage = "impact_closure"
        callers: list[dict[str, str]] = []
        for item in resolved_callers:
            callers.append(_normalize_caller(item))

        arity_increasing = [d for d in deltas if d.arity_increased]
        if auto_discover_callers and arity_increasing:
            for delta in arity_increasing:
                for site in find_call_sites(base_sources, delta.symbol):
                    # Exclude the defining site itself.
                    if site["path"] == delta.path and site.get(
                        "symbol"
                    ) == delta.symbol:
                        # call sites use caller function name in path::caller
                        pass
                    callers.append(site)
                # Also search candidate sources for residual call sites.
                for site in find_call_sites(candidate_sources, delta.symbol):
                    callers.append(site)

        # Deduplicate callers by id.
        by_id: dict[str, dict[str, str]] = {}
        for caller in callers:
            by_id[caller["caller_id"]] = caller
        callers = list(by_id.values())
        completed.append(stage)

        stage = "consumer_frontier"
        unknown = tuple(
            sorted({str(u).strip() for u in unknown_frontier if str(u).strip()})
        )
        if (
            self.policy.require_unknown_frontier_abstain
            and unknown
            and arity_increasing
        ):
            completed.append(stage)
            overlay = CandidateOverlayReceipt(
                overlay_id=overlay_id,
                proposal_id=proposal_id,
                repository_id=repository_id,
                base_tree_id=base_tree_id,
                candidate_tree_id=candidate_tree_id,
                changed_paths=tuple(
                    sorted(set(base_sources) | set(candidate_sources))
                ),
                write_set=write_set_t,
                signature_deltas=deltas,
                resolved_callers=tuple(c["caller_id"] for c in callers),
                omitted_callers=(),
                unknown_frontier=unknown,
                caller_dispositions=tuple(
                    CallerDispositionRecord(
                        caller_id=c["caller_id"],
                        path=c.get("path", ""),
                        symbol=c.get("symbol", ""),
                        disposition=OverlayCallerDisposition.UNKNOWN,
                        detail="required unknown frontier abstains",
                    )
                    for c in callers
                ),
                impact_closure_id=impact_closure_id or f"impact:{overlay_id[:16]}",
                consumer_frontier_id=(
                    consumer_frontier_id or f"frontier:{overlay_id[:16]}"
                ),
                delta_id=delta_id,
                mutation_allowed=False,
            )
            completed.append("caller_disposition")
            completed.append("admit_or_reject")
            return CandidateOverlayGateResult(
                disposition=OverlayGateDisposition.ABSTAINED,
                detail=(
                    "required unknown frontier present; "
                    "abstaining before mutation"
                ),
                reason_codes=("unknown_frontier_required",),
                stages_completed=tuple(completed),
                overlay=overlay,
                mutation_allowed=False,
            )
        completed.append(stage)

        # 5. Disposition every resolved caller.
        stage = "caller_disposition"
        compat = {
            str(x).strip() for x in compatibility_proofs if str(x).strip()
        }
        no_change = {
            str(x).strip() for x in no_change_proofs if str(x).strip()
        }
        write_paths = set(write_set_t)
        dispositions: list[CallerDispositionRecord] = []
        omitted: list[str] = []
        for caller in sorted(callers, key=lambda c: c["caller_id"]):
            cid = caller["caller_id"]
            path = caller.get("path", "")
            if cid in no_change or path in no_change:
                dispositions.append(
                    CallerDispositionRecord(
                        caller_id=cid,
                        path=path,
                        symbol=caller.get("symbol", ""),
                        disposition=OverlayCallerDisposition.NO_CHANGE_PROOF,
                        detail="explicit no-change proof",
                    )
                )
                continue
            if cid in compat or path in compat:
                dispositions.append(
                    CallerDispositionRecord(
                        caller_id=cid,
                        path=path,
                        symbol=caller.get("symbol", ""),
                        disposition=(
                            OverlayCallerDisposition.COMPATIBILITY_PROOF
                        ),
                        detail="explicit compatibility proof",
                    )
                )
                continue
            if path and path in write_paths:
                dispositions.append(
                    CallerDispositionRecord(
                        caller_id=cid,
                        path=path,
                        symbol=caller.get("symbol", ""),
                        disposition=OverlayCallerDisposition.IN_WRITE_SET,
                        detail="caller path already in write set",
                    )
                )
                continue
            if cid in write_paths:
                dispositions.append(
                    CallerDispositionRecord(
                        caller_id=cid,
                        path=path,
                        symbol=caller.get("symbol", ""),
                        disposition=OverlayCallerDisposition.IN_WRITE_SET,
                        detail="caller id already in write set",
                    )
                )
                continue
            # Only arity-increasing (or any) signature changes require caller
            # coverage when there is a resolved caller of a changed symbol.
            changed_symbols = {d.symbol for d in deltas if d.changed}
            if caller.get("symbol") in changed_symbols or any(
                d.arity_increased for d in deltas
            ):
                omitted.append(cid)
                dispositions.append(
                    CallerDispositionRecord(
                        caller_id=cid,
                        path=path,
                        symbol=caller.get("symbol", ""),
                        disposition=OverlayCallerDisposition.OMITTED,
                        detail="caller omitted from existing write set",
                    )
                )
            else:
                dispositions.append(
                    CallerDispositionRecord(
                        caller_id=cid,
                        path=path,
                        symbol=caller.get("symbol", ""),
                        disposition=OverlayCallerDisposition.IN_WRITE_SET,
                        detail="no signature change requires caller",
                    )
                )
        completed.append(stage)

        # 6. Admit / reject / expand.
        stage = "admit_or_reject"
        expanded: list[str] = list(write_set_t)
        if omitted and arity_increasing:
            if (
                self.policy.expand_write_set_on_omission
                and self.policy.reject_omitted_callers
            ):
                # Expand then re-admit.
                for caller in callers:
                    if caller["caller_id"] in omitted:
                        path = caller.get("path") or ""
                        if path and path not in expanded:
                            expanded.append(path)
                        # Mark expanded disposition.
                        dispositions = [
                            (
                                CallerDispositionRecord(
                                    caller_id=d.caller_id,
                                    path=d.path,
                                    symbol=d.symbol,
                                    disposition=OverlayCallerDisposition.EXPANDED,
                                    detail="write set expanded to include caller",
                                )
                                if d.caller_id in omitted
                                else d
                            )
                            for d in dispositions
                        ]
                completed.append(stage)
                overlay = CandidateOverlayReceipt(
                    overlay_id=overlay_id,
                    proposal_id=proposal_id,
                    repository_id=repository_id,
                    base_tree_id=base_tree_id,
                    candidate_tree_id=candidate_tree_id,
                    changed_paths=tuple(
                        sorted(set(base_sources) | set(candidate_sources))
                    ),
                    write_set=write_set_t,
                    signature_deltas=deltas,
                    resolved_callers=tuple(c["caller_id"] for c in callers),
                    omitted_callers=tuple(sorted(omitted)),
                    unknown_frontier=unknown,
                    caller_dispositions=tuple(dispositions),
                    expanded_write_set=tuple(sorted(set(expanded))),
                    impact_closure_id=(
                        impact_closure_id or f"impact:{overlay_id[:16]}"
                    ),
                    consumer_frontier_id=(
                        consumer_frontier_id or f"frontier:{overlay_id[:16]}"
                    ),
                    delta_id=delta_id,
                    mutation_allowed=True,
                )
                return CandidateOverlayGateResult(
                    disposition=OverlayGateDisposition.EXPANDED,
                    detail=(
                        "signature change omitted callers; write set expanded "
                        "for re-admission"
                    ),
                    reason_codes=(
                        "omitted_callers_expanded",
                        "signature_arity_increase",
                    ),
                    stages_completed=tuple(completed),
                    overlay=overlay,
                    mutation_allowed=True,
                    expanded_write_set=tuple(sorted(set(expanded))),
                )
            if self.policy.reject_omitted_callers:
                completed.append(stage)
                overlay = CandidateOverlayReceipt(
                    overlay_id=overlay_id,
                    proposal_id=proposal_id,
                    repository_id=repository_id,
                    base_tree_id=base_tree_id,
                    candidate_tree_id=candidate_tree_id,
                    changed_paths=tuple(
                        sorted(set(base_sources) | set(candidate_sources))
                    ),
                    write_set=write_set_t,
                    signature_deltas=deltas,
                    resolved_callers=tuple(c["caller_id"] for c in callers),
                    omitted_callers=tuple(sorted(omitted)),
                    unknown_frontier=unknown,
                    caller_dispositions=tuple(dispositions),
                    impact_closure_id=(
                        impact_closure_id or f"impact:{overlay_id[:16]}"
                    ),
                    consumer_frontier_id=(
                        consumer_frontier_id or f"frontier:{overlay_id[:16]}"
                    ),
                    delta_id=delta_id,
                    mutation_allowed=False,
                )
                return CandidateOverlayGateResult(
                    disposition=OverlayGateDisposition.REJECTED,
                    detail=(
                        "signature change omits resolved callers from write set"
                    ),
                    reason_codes=(
                        "omitted_callers",
                        "signature_arity_increase",
                    ),
                    stages_completed=tuple(completed),
                    overlay=overlay,
                    mutation_allowed=False,
                )
            completed.append(stage)
            overlay = CandidateOverlayReceipt(
                overlay_id=overlay_id,
                proposal_id=proposal_id,
                repository_id=repository_id,
                base_tree_id=base_tree_id,
                candidate_tree_id=candidate_tree_id,
                changed_paths=tuple(
                    sorted(set(base_sources) | set(candidate_sources))
                ),
                write_set=write_set_t,
                signature_deltas=deltas,
                resolved_callers=tuple(c["caller_id"] for c in callers),
                omitted_callers=tuple(sorted(omitted)),
                unknown_frontier=unknown,
                caller_dispositions=tuple(dispositions),
                impact_closure_id=(
                    impact_closure_id or f"impact:{overlay_id[:16]}"
                ),
                consumer_frontier_id=(
                    consumer_frontier_id or f"frontier:{overlay_id[:16]}"
                ),
                delta_id=delta_id,
                mutation_allowed=False,
            )
            return CandidateOverlayGateResult(
                disposition=OverlayGateDisposition.DEFERRED,
                detail="omitted callers deferred for expansion/re-admission",
                reason_codes=("omitted_callers_deferred",),
                stages_completed=tuple(completed),
                overlay=overlay,
                mutation_allowed=False,
            )

        completed.append(stage)
        overlay = CandidateOverlayReceipt(
            overlay_id=overlay_id,
            proposal_id=proposal_id,
            repository_id=repository_id,
            base_tree_id=base_tree_id,
            candidate_tree_id=candidate_tree_id,
            changed_paths=tuple(
                sorted(set(base_sources) | set(candidate_sources))
            ),
            write_set=write_set_t,
            signature_deltas=deltas,
            resolved_callers=tuple(c["caller_id"] for c in callers),
            omitted_callers=(),
            unknown_frontier=unknown,
            caller_dispositions=tuple(dispositions),
            expanded_write_set=write_set_t,
            impact_closure_id=impact_closure_id or f"impact:{overlay_id[:16]}",
            consumer_frontier_id=(
                consumer_frontier_id or f"frontier:{overlay_id[:16]}"
            ),
            delta_id=delta_id,
            mutation_allowed=True,
        )
        return CandidateOverlayGateResult(
            disposition=OverlayGateDisposition.ADMITTED,
            detail="all resolved callers dispositioned; mutation may proceed",
            reason_codes=("callers_complete",),
            stages_completed=tuple(completed),
            overlay=overlay,
            mutation_allowed=True,
            expanded_write_set=write_set_t,
        )


# ---------------------------------------------------------------------------
# Live logic-repair controller
# ---------------------------------------------------------------------------


class LiveLogicRepairController:
    """Versioned feature-gated edge path for live logic repair.

    Invokes the full static facts → admitted-plan chain and intercepts
    ordinary provider proposals as read-only overlays before mutation.
    """

    INTERFACE: Final[str] = LIVE_LOGIC_REPAIR_CONTROLLER_INTERFACE
    VERSION: Final[int] = LIVE_LOGIC_REPAIR_CONTROLLER_VERSION

    def __init__(
        self,
        policy: LiveLogicRepairPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        self.policy = LiveLogicRepairPolicy.from_value(policy)
        self._overlay_gate = CandidateOverlayContractDeltaGate(self.policy)

    def run(
        self,
        request: LiveLogicRepairRequest | Mapping[str, Any],
    ) -> LiveLogicRepairResult:
        if not isinstance(request, LiveLogicRepairRequest):
            if not isinstance(request, Mapping):
                raise TypeError(
                    "live logic repair request must be a mapping or "
                    "LiveLogicRepairRequest"
                )
            request = LiveLogicRepairRequest.from_mapping(request)

        mode = request.mode
        if isinstance(mode, LiveLogicRepairMode):
            mode_value = mode.value
        else:
            mode_value = str(mode)

        if not self.policy.enable_live_logic_repair:
            return _disabled_result(mode=mode_value)

        if mode is LiveLogicRepairMode.PROPOSAL_OVERLAY or mode_value == (
            LiveLogicRepairMode.PROPOSAL_OVERLAY.value
        ):
            return self._run_proposal_overlay(request)

        if mode is LiveLogicRepairMode.CHANGE_PROPAGATION or mode_value == (
            LiveLogicRepairMode.CHANGE_PROPAGATION.value
        ):
            return self._run_change_propagation(request)

        return self._run_contract_repair(request)

    def intercept_proposal(
        self,
        *,
        proposal_id: str,
        repository_id: str,
        base_tree_id: str,
        candidate_tree_id: str,
        write_set: Sequence[str],
        base_sources: Mapping[str, str],
        candidate_sources: Mapping[str, str],
        resolved_callers: Sequence[Mapping[str, Any] | str] = (),
        unknown_frontier: Sequence[str] = (),
        compatibility_proofs: Sequence[str] = (),
        no_change_proofs: Sequence[str] = (),
        impact_closure_id: str = "",
        consumer_frontier_id: str = "",
    ) -> CandidateOverlayGateResult:
        """Public entry: intercept one ordinary provider proposal."""

        return self._overlay_gate.evaluate(
            proposal_id=proposal_id,
            repository_id=repository_id,
            base_tree_id=base_tree_id,
            candidate_tree_id=candidate_tree_id,
            write_set=write_set,
            base_sources=base_sources,
            candidate_sources=candidate_sources,
            resolved_callers=resolved_callers,
            unknown_frontier=unknown_frontier,
            compatibility_proofs=compatibility_proofs,
            no_change_proofs=no_change_proofs,
            impact_closure_id=impact_closure_id,
            consumer_frontier_id=consumer_frontier_id,
        )

    def bridge_proof_bundle(
        self,
        *,
        candidate_id: str,
        repository_id: str,
        tree_id: str,
        prediction_decision: Any = None,
        prediction_receipts: Sequence[Any] = (),
        base_proof_bundle: Any = None,
    ) -> Any:
        """Public entry: bridge predictions into CandidateProofBundle."""

        return bridge_predictions_into_proof_bundle(
            candidate_id=candidate_id,
            repository_id=repository_id,
            tree_id=tree_id,
            prediction_decision=prediction_decision,
            prediction_receipts=prediction_receipts,
            base_proof_bundle=base_proof_bundle,
        )

    # ------------------------------------------------------------------
    # Mode runners
    # ------------------------------------------------------------------

    def _run_proposal_overlay(
        self, request: LiveLogicRepairRequest
    ) -> LiveLogicRepairResult:
        mode = LiveLogicRepairMode.PROPOSAL_OVERLAY.value
        proposal_id = request.proposal_id or str(
            getattr(request.proposal, "proposal_id", "") or "proposal:overlay"
        )
        gate = self._overlay_gate.evaluate(
            proposal_id=proposal_id,
            repository_id=request.repository_id,
            base_tree_id=request.tree_id,
            candidate_tree_id=str(
                getattr(request.roots, "candidate_tree_id", "") or request.tree_id
            ),
            write_set=request.write_set,
            base_sources=request.base_sources,
            candidate_sources=request.candidate_sources,
            resolved_callers=request.resolved_callers,
            unknown_frontier=request.unknown_frontier,
            compatibility_proofs=request.compatibility_proofs,
            no_change_proofs=request.no_change_proofs,
        )
        disposition_map = {
            OverlayGateDisposition.ADMITTED: (
                LiveLogicRepairDisposition.ADMITTED.value
            ),
            OverlayGateDisposition.EXPANDED: (
                LiveLogicRepairDisposition.EXPANDED.value
            ),
            OverlayGateDisposition.REJECTED: (
                LiveLogicRepairDisposition.REJECTED.value
            ),
            OverlayGateDisposition.DEFERRED: (
                LiveLogicRepairDisposition.DEFERRED.value
            ),
            OverlayGateDisposition.ABSTAINED: (
                LiveLogicRepairDisposition.ABSTAINED.value
            ),
            OverlayGateDisposition.DISABLED: (
                LiveLogicRepairDisposition.DISABLED.value
            ),
        }
        write_paths = gate.expanded_write_set or tuple(request.write_set)
        return LiveLogicRepairResult(
            enabled=True,
            mode=mode,
            stage=gate.stages_completed[-1] if gate.stages_completed else "overlay",
            disposition=disposition_map.get(
                gate.disposition, LiveLogicRepairDisposition.REJECTED.value
            ),
            detail=gate.detail,
            provider_invoked=False,
            stages_completed=gate.stages_completed,
            reason_codes=gate.reason_codes,
            overlay_gate=gate,
            write_paths=write_paths,
            mutation_allowed=gate.mutation_allowed,
        )

    def _run_logic_stages(
        self,
        request: LiveLogicRepairRequest,
        *,
        mode: str,
        prefix_stages: Sequence[str],
        final_stage: str,
    ) -> tuple[list[str], Any, LiveLogicRepairResult | None]:
        """Run shared goal→admission chain; return (completed, bundle, early)."""

        completed: list[str] = list(prefix_stages)
        callbacks = request.stage_callbacks

        # Optional stages may be recorded when a prediction decision is already
        # bound (hermetic analytical path) or when a pure callback is supplied.
        optional_when_prediction = frozenset(
            {
                "corpus",
                "tactician",
                "hypothesis",
                "gate",
                "lowering",
                "hammer",
                "refinement",
            }
        )

        stage_values: list[tuple[str, Any]] = [
            ("goal", request.goals),
            ("corpus", request.corpus),
            ("tactician", request.tactician_plan),
            ("hypothesis", request.hypotheses),
            ("gate", request.plan_gate_receipt),
            ("lowering", request.lowering),
            ("hammer", request.hammer_receipt),
            ("refinement", request.refinement),
            ("admission", request.prediction_decision),
        ]

        for name, value in stage_values:
            if name in callbacks:
                try:
                    callbacks[name](request)
                except Exception as exc:  # fail-closed
                    return (
                        completed,
                        None,
                        _fail(
                            mode=mode,
                            stage=name,
                            disposition=(
                                LiveLogicRepairDisposition.REJECTED.value
                            ),
                            detail=f"stage {name} callback failed: {exc}",
                            completed=completed,
                            reason_codes=(f"{name}_failed",),
                        ),
                    )
                completed.append(name)
                continue

            missing = value is None or (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
                and len(value) == 0
                and name not in {"goal", "hypothesis", "refinement"}
            )
            # Empty goal/hypothesis/refinement sequences are valid when the
            # stage is intentionally empty under an admitted analytical path.
            empty_seq = (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
                and len(value) == 0
            )
            if value is None and name == "admission":
                if request.analytical_success and (
                    request.target_admission is not None
                    or request.atomic_plan_admission is not None
                    or request.evidence_bundle is not None
                ):
                    completed.append(name)
                    continue
                return (
                    completed,
                    None,
                    _fail(
                        mode=mode,
                        stage=name,
                        disposition=(
                            LiveLogicRepairDisposition.ABSTAINED.value
                        ),
                        detail=f"stage {name} produced no bound artifact",
                        completed=completed,
                        reason_codes=(f"missing_{name}",),
                    ),
                )
            if value is None or (missing and not empty_seq):
                if (
                    name in optional_when_prediction
                    and request.prediction_decision is not None
                ):
                    completed.append(name)
                    continue
                if name == "goal" and empty_seq:
                    completed.append(name)
                    continue
                return (
                    completed,
                    None,
                    _fail(
                        mode=mode,
                        stage=name,
                        disposition=(
                            LiveLogicRepairDisposition.ABSTAINED.value
                        ),
                        detail=f"stage {name} produced no bound artifact",
                        completed=completed,
                        reason_codes=(f"missing_{name}",),
                    ),
                )
            if empty_seq and name in {"goal", "hypothesis", "refinement"}:
                completed.append(name)
                continue
            if value is None:
                return (
                    completed,
                    None,
                    _fail(
                        mode=mode,
                        stage=name,
                        disposition=(
                            LiveLogicRepairDisposition.ABSTAINED.value
                        ),
                        detail=f"stage {name} produced no bound artifact",
                        completed=completed,
                        reason_codes=(f"missing_{name}",),
                    ),
                )
            completed.append(name)

        # Bridge predictions into CandidateProofBundle before target/atomic
        # plan admission.
        try:
            bundle = bridge_predictions_into_proof_bundle(
                candidate_id=str(
                    getattr(request.prediction_decision, "content_id", "")
                    or f"candidate:{request.task_id or request.tree_id}"
                ),
                repository_id=request.repository_id,
                tree_id=request.tree_id,
                prediction_decision=request.prediction_decision,
                prediction_receipts=request.prediction_receipts,
                base_proof_bundle=request.base_proof_bundle,
            )
        except Exception as exc:
            return (
                completed,
                None,
                _fail(
                    mode=mode,
                    stage="admission",
                    disposition=LiveLogicRepairDisposition.REJECTED.value,
                    detail=f"prediction bridge failed: {exc}",
                    completed=completed,
                    reason_codes=("prediction_bridge_failed",),
                ),
            )

        # Unknown frontier abstain before plan admission.
        if (
            self.policy.require_unknown_frontier_abstain
            and request.unknown_frontier
        ):
            return (
                completed,
                bundle,
                _fail(
                    mode=mode,
                    stage=final_stage,
                    disposition=LiveLogicRepairDisposition.ABSTAINED.value,
                    detail=(
                        "required unknown frontier abstains before admission"
                    ),
                    completed=completed,
                    reason_codes=("unknown_frontier_required",),
                    proof_bundle=bundle,
                    prediction_decision=request.prediction_decision,
                ),
            )

        return completed, bundle, None

    def _run_contract_repair(
        self, request: LiveLogicRepairRequest
    ) -> LiveLogicRepairResult:
        mode = LiveLogicRepairMode.CONTRACT_REPAIR.value
        completed: list[str] = []
        callbacks = request.stage_callbacks

        # Prefix: trace / contracts / retrieval
        for name, value in (
            ("trace", request.trace),
            ("contracts", request.contracts),
            ("retrieval", request.candidates),
        ):
            if name in callbacks:
                try:
                    callbacks[name](request)
                except Exception as exc:
                    return _fail(
                        mode=mode,
                        stage=name,
                        disposition=LiveLogicRepairDisposition.REJECTED.value,
                        detail=f"stage {name} failed: {exc}",
                        completed=completed,
                        reason_codes=(f"{name}_failed",),
                    )
            if value is None or (
                name == "retrieval"
                and isinstance(value, Sequence)
                and not value
            ):
                return _fail(
                    mode=mode,
                    stage=name,
                    disposition=LiveLogicRepairDisposition.REJECTED.value,
                    detail=f"stage {name} produced no bound artifact",
                    completed=completed,
                    reason_codes=(f"missing_{name}",),
                )
            completed.append(name)

        completed, bundle, early = self._run_logic_stages(
            request,
            mode=mode,
            prefix_stages=completed,
            final_stage="target_admission",
        )
        if early is not None:
            return early

        stage = "target_admission"
        if request.target_admission is None and "target_admission" not in callbacks:
            return _fail(
                mode=mode,
                stage=stage,
                disposition=LiveLogicRepairDisposition.ABSTAINED.value,
                detail="target admission missing after logic stages",
                completed=completed,
                reason_codes=("missing_target_admission",),
                proof_bundle=bundle,
                prediction_decision=request.prediction_decision,
            )
        if "target_admission" in callbacks:
            try:
                callbacks["target_admission"](request)
            except Exception as exc:
                return _fail(
                    mode=mode,
                    stage=stage,
                    disposition=LiveLogicRepairDisposition.REJECTED.value,
                    detail=f"target admission failed: {exc}",
                    completed=completed,
                    reason_codes=("target_admission_failed",),
                    proof_bundle=bundle,
                )
        completed.append(stage)

        provider_invoked = False
        model_overlay = None
        if request.model_required and not request.analytical_success:
            if not self.policy.allow_provider_for_model_steps:
                return _fail(
                    mode=mode,
                    stage=stage,
                    disposition=LiveLogicRepairDisposition.REJECTED.value,
                    detail="model steps disabled by policy",
                    completed=completed,
                    reason_codes=("model_steps_disabled",),
                    proof_bundle=bundle,
                )
            # Model calls use only LPR-016 overlays projected into existing
            # packets — never a free-form provider prompt.
            model_overlay = self._materialize_lpr016_overlay(request)
            if model_overlay is None:
                return _fail(
                    mode=mode,
                    stage=stage,
                    disposition=LiveLogicRepairDisposition.ABSTAINED.value,
                    detail="LPR-016 context overlay materialization abstained",
                    completed=completed,
                    reason_codes=("lpr016_overlay_abstained",),
                    proof_bundle=bundle,
                )
            provider_invoked = True
        elif request.analytical_success:
            # Hard invariant: analytical success makes no model call.
            provider_invoked = False

        if (
            self.policy.revalidate_roots_and_receipts
            and request.roots is not None
            and request.logic_roots is not None
        ):
            # Identity presence only at this edge; detailed gates revalidate.
            if not (
                getattr(request.roots, "repository_id", None)
                or (isinstance(request.roots, Mapping) and request.roots.get("repository_id"))
            ):
                return _fail(
                    mode=mode,
                    stage=stage,
                    disposition=LiveLogicRepairDisposition.REJECTED.value,
                    detail="roots revalidation failed",
                    completed=completed,
                    reason_codes=("root_revalidation_failed",),
                    proof_bundle=bundle,
                    provider_invoked=provider_invoked,
                )

        write_paths = tuple(request.scope_paths) or tuple(request.write_set)
        return LiveLogicRepairResult(
            enabled=True,
            mode=mode,
            stage=stage,
            disposition=LiveLogicRepairDisposition.ADMITTED.value,
            detail=(
                "logic stages completed; target admission ready; "
                "proof bundle bridged"
            ),
            provider_invoked=provider_invoked,
            stages_completed=tuple(completed),
            reason_codes=("admitted",),
            proof_bundle=bundle,
            prediction_decision=request.prediction_decision,
            target_admission=request.target_admission,
            write_paths=write_paths,
            model_context_overlay=model_overlay,
            mutation_allowed=False,  # mutation still requires transaction path
        )

    def _run_change_propagation(
        self, request: LiveLogicRepairRequest
    ) -> LiveLogicRepairResult:
        mode = LiveLogicRepairMode.CHANGE_PROPAGATION.value
        completed: list[str] = []
        callbacks = request.stage_callbacks

        for name, value in (
            ("delta", request.delta),
            ("graph", request.graph_id or request.roots),
            ("impact", request.impact_closure),
            ("consumer", request.consumers),
            ("value", request.value_proofs),
            ("behavior", request.behavior_gaps),
        ):
            if name in callbacks:
                try:
                    callbacks[name](request)
                except Exception as exc:
                    return _fail(
                        mode=mode,
                        stage=name,
                        disposition=LiveLogicRepairDisposition.REJECTED.value,
                        detail=f"stage {name} failed: {exc}",
                        completed=completed,
                        reason_codes=(f"{name}_failed",),
                    )
            missing = value is None or (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
                and not value
                and name in {"consumer", "value", "behavior"}
            )
            if missing and name not in callbacks:
                # value/behavior may be empty when analytical transforms carry
                # the proof; require impact at minimum.
                if name in {"delta", "graph", "impact"}:
                    return _fail(
                        mode=mode,
                        stage=name,
                        disposition=LiveLogicRepairDisposition.REJECTED.value,
                        detail=f"stage {name} produced no bound artifact",
                        completed=completed,
                        reason_codes=(f"missing_{name}",),
                    )
            completed.append(name)

        completed, bundle, early = self._run_logic_stages(
            request,
            mode=mode,
            prefix_stages=completed,
            final_stage="atomic_plan_admission",
        )
        if early is not None:
            return early

        stage = "atomic_plan_admission"
        if (
            request.atomic_plan_admission is None
            and request.evidence_bundle is None
            and "atomic_plan_admission" not in callbacks
        ):
            return _fail(
                mode=mode,
                stage=stage,
                disposition=LiveLogicRepairDisposition.ABSTAINED.value,
                detail="atomic plan admission missing after logic stages",
                completed=completed,
                reason_codes=("missing_atomic_plan_admission",),
                proof_bundle=bundle,
                prediction_decision=request.prediction_decision,
            )
        if "atomic_plan_admission" in callbacks:
            try:
                callbacks["atomic_plan_admission"](request)
            except Exception as exc:
                return _fail(
                    mode=mode,
                    stage=stage,
                    disposition=LiveLogicRepairDisposition.REJECTED.value,
                    detail=f"atomic plan admission failed: {exc}",
                    completed=completed,
                    reason_codes=("atomic_plan_failed",),
                    proof_bundle=bundle,
                )
        completed.append(stage)

        provider_invoked = False
        model_overlay = None
        if request.model_required and not request.analytical_success:
            model_overlay = self._materialize_lpr016_overlay(request)
            if model_overlay is None:
                return _fail(
                    mode=mode,
                    stage=stage,
                    disposition=LiveLogicRepairDisposition.ABSTAINED.value,
                    detail="LPR-016 context overlay materialization abstained",
                    completed=completed,
                    reason_codes=("lpr016_overlay_abstained",),
                    proof_bundle=bundle,
                )
            provider_invoked = True

        write_paths = tuple(request.scope_paths) or tuple(request.write_set)
        return LiveLogicRepairResult(
            enabled=True,
            mode=mode,
            stage=stage,
            disposition=LiveLogicRepairDisposition.ADMITTED.value,
            detail=(
                "logic stages completed; atomic plan admission ready; "
                "proof bundle bridged"
            ),
            provider_invoked=provider_invoked,
            stages_completed=tuple(completed),
            reason_codes=("admitted",),
            proof_bundle=bundle,
            prediction_decision=request.prediction_decision,
            atomic_plan_admission=request.atomic_plan_admission,
            write_paths=write_paths,
            model_context_overlay=model_overlay,
            mutation_allowed=False,
        )

    def _materialize_lpr016_overlay(
        self, request: LiveLogicRepairRequest
    ) -> Any:
        """Materialize model context only through LPR-016 existing packets."""

        materialize = request.stage_callbacks.get("lpr016_materialize")
        if callable(materialize):
            try:
                return materialize(request)
            except Exception:
                return None
        # Without an explicit materializer callback, abstain rather than
        # invent a free-form provider prompt.
        return None


def run_live_logic_repair(
    request: LiveLogicRepairRequest | Mapping[str, Any],
    *,
    policy: LiveLogicRepairPolicy | Mapping[str, Any] | None = None,
) -> LiveLogicRepairResult:
    """Module entry point matching :meth:`LiveLogicRepairController.run`."""

    return LiveLogicRepairController(policy=policy).run(request)


def daemon_execute_live_logic_repair(
    daemon: Any,
    request: Any,
    *,
    policy: Any = None,
    enable: bool = True,
) -> Any:
    """Daemon-facing entry: run the gated controller (lazy-safe, thin host)."""

    controller_policy = LiveLogicRepairPolicy(
        enable_live_logic_repair=bool(enable),
    )
    if policy is not None:
        if isinstance(policy, LiveLogicRepairPolicy):
            controller_policy = policy
            if enable and not controller_policy.enable_live_logic_repair:
                controller_policy = LiveLogicRepairPolicy.from_value(
                    {
                        **controller_policy.to_dict(),
                        "enable_live_logic_repair": True,
                    }
                )
        elif isinstance(policy, Mapping):
            merged = {
                **controller_policy.to_dict(),
                **dict(policy),
                "enable_live_logic_repair": bool(enable),
            }
            controller_policy = LiveLogicRepairPolicy.from_value(merged)
    controller = LiveLogicRepairController(policy=controller_policy)
    if not isinstance(request, LiveLogicRepairRequest):
        if isinstance(request, Mapping):
            request = LiveLogicRepairRequest.from_mapping(request)
    result = controller.run(request)
    try:
        record = getattr(daemon, "_record_event", None)
        if callable(record):
            record(
                "live_logic_repair",
                {
                    "enabled": result.enabled,
                    "mode": result.mode,
                    "stage": result.stage,
                    "disposition": result.disposition,
                    "detail": result.detail,
                    "provider_invoked": result.provider_invoked,
                    "write_paths": list(result.write_paths),
                    "mutation_allowed": result.mutation_allowed,
                    "completion_authoritative": False,
                },
            )
    except Exception:
        pass
    return result


def daemon_intercept_logic_repair_proposal(
    daemon: Any,
    **kw: Any,
) -> Any:
    """Daemon-facing entry: read-only candidate-overlay contract delta gate."""

    _ = daemon  # host is reserved for future event recording
    enable = bool(kw.pop("enable", True))
    expand = bool(kw.pop("expand_write_set_on_omission", True))
    policy = LiveLogicRepairPolicy(
        enable_live_logic_repair=enable,
        expand_write_set_on_omission=expand,
    )
    return LiveLogicRepairController(policy=policy).intercept_proposal(**kw)


def daemon_assert_no_logic_repair_write_bypass(
    *,
    write_performed: bool,
    overlay_mutation_allowed: bool,
    transaction_committed: bool = False,
) -> None:
    """Fail closed when a logic-repair write would bypass overlay + transaction."""

    if write_performed and not overlay_mutation_allowed:
        raise RuntimeError(
            "live logic-repair write cannot bypass CandidateOverlayContractDeltaGate"
        )
    if write_performed and overlay_mutation_allowed and not transaction_committed:
        # Overlay admission is necessary but not sufficient; mutations still
        # require the existing transaction path.
        raise RuntimeError(
            "live logic-repair write cannot bypass ChangePropagationTransaction"
        )


__all__ = [
    "CHANGE_PROPAGATION_STAGE_ORDER",
    "CONTRACT_REPAIR_STAGE_ORDER",
    "PROPOSAL_OVERLAY_STAGE_ORDER",
    "LIVE_LOGIC_REPAIR_CONTROLLER_INTERFACE",
    "LIVE_LOGIC_REPAIR_CONTROLLER_SCHEMA",
    "LIVE_LOGIC_REPAIR_CONTROLLER_VERSION",
    "PRODUCER_ID",
    "CallableSignatureDelta",
    "CallerDispositionRecord",
    "CandidateOverlayContractDeltaGate",
    "CandidateOverlayGateResult",
    "CandidateOverlayReceipt",
    "LiveLogicRepairController",
    "LiveLogicRepairDisposition",
    "LiveLogicRepairError",
    "LiveLogicRepairMode",
    "LiveLogicRepairPolicy",
    "LiveLogicRepairRequest",
    "LiveLogicRepairResult",
    "OverlayCallerDisposition",
    "OverlayGateDisposition",
    "bridge_predictions_into_proof_bundle",
    "compute_signature_deltas",
    "daemon_assert_no_logic_repair_write_bypass",
    "daemon_execute_live_logic_repair",
    "daemon_intercept_logic_repair_proposal",
    "extract_python_signatures",
    "find_call_sites",
    "run_live_logic_repair",
]
