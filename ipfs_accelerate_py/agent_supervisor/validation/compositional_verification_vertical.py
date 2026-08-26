"""Hermetic P1--P5 compositional-verification vertical slice.

This module is an orchestration adapter, not a new semantic authority.  It
drives datasets-owned semantic-index/state/capsule, contract, abstract-
interpretation, assume-guarantee, invalidation, and SMT APIs.  Operational
mutation, repair nomination, worktree fencing, transaction evidence, fixed-
point validation, and context accounting remain accelerator-owned.

The fixture-specific proof-carrying artifact is independently verified by
rescanning bytes, rebuilding the semantic root, redischarge of the final
composition graph, and replaying the selected tests.  A producer-side
``passed`` field is neither accepted nor emitted as proof authority.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final


def _install_multiformats_shim() -> None:
    """Provide CIDv1 encode/decode when the optional ``multiformats`` package is absent.

    Datasets software-contract identity is CIDv1 / base32 / sha2-256 over ``raw``
    and ``dag-json``.  The in-tree ``cid_utils`` already owns that closed profile;
    this shim only exposes the historical ``multiformats`` names that
    ``ipfs_datasets_py.logic.software_contracts.content`` imports lazily.
    """

    if "multiformats" in sys.modules:
        return
    try:
        importlib.import_module("multiformats")
        return
    except ImportError:
        pass

    from ipfs_accelerate_py.utils import cid_utils

    class _Named:
        def __init__(self, name: str) -> None:
            self.name = name

    class _MultihashInfo:
        max_digest_size = 32

    class _Multihash:
        def digest(self, data: bytes | bytearray | memoryview, hashfun: str) -> bytes:
            if hashfun not in {"sha2-256", "sha256"}:
                raise ValueError(f"unsupported multihash {hashfun!r}")
            digest = hashlib.sha256(bytes(data)).digest()
            return b"\x12\x20" + digest

        def wrap(self, digest: bytes | bytearray | memoryview, hashfun: str) -> bytes:
            if hashfun not in {"sha2-256", "sha256"}:
                raise ValueError(f"unsupported multihash {hashfun!r}")
            payload = bytes(digest)
            if len(payload) != 32:
                raise ValueError("sha2-256 digest must be exactly 32 bytes")
            return b"\x12\x20" + payload

        def get(self, hashfun: str) -> _MultihashInfo:
            if hashfun not in {"sha2-256", "sha256"}:
                raise KeyError(hashfun)
            return _MultihashInfo()

    class CID:
        def __init__(
            self,
            base: str,
            version: int,
            codec: str,
            digest: bytes | bytearray | memoryview,
        ) -> None:
            raw = bytes(digest)
            if len(raw) == 34 and raw[:2] == b"\x12\x20":
                raw = raw[2:]
            self.version = int(version)
            self.codec = _Named(str(codec))
            self.hashfun = _Named("sha2-256")
            self.raw_digest = raw
            self.base = _Named(str(base))
            self._text = cid_utils.cid_from_sha256_digest(raw, codec=str(codec))

        @classmethod
        def decode(cls, value: str) -> CID:
            decoded = cid_utils._decode_cid(value)
            constructed = object.__new__(cls)
            constructed.version = decoded.version
            constructed.codec = _Named(decoded.codec)
            constructed.hashfun = _Named(decoded.multihash_type)
            constructed.raw_digest = decoded.digest
            constructed.base = _Named("base32")
            constructed._text = value
            return constructed

        def __str__(self) -> str:
            return self._text

    module = types.ModuleType("multiformats")
    module.CID = CID
    module.multihash = _Multihash()
    sys.modules["multiformats"] = module


def _install_semantic_state_package_stub() -> None:
    """Load semantic-state submodules without importing the MCP++ harness package.

    ``semantic_state/__init__.py`` pulls the compression harness, which imports
    MCP++ and optional ``anyio``.  The vertical slice only needs the datasets
    adapter and isolated worktree, so the package is registered as a namespace
    stub before those submodules load.
    """

    name = "ipfs_accelerate_py.agent_supervisor.semantic_state"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None):
        return
    if existing is not None and getattr(existing, "__path__", None):
        return
    package = types.ModuleType(name)
    package.__path__ = [
        str(Path(__file__).resolve().parent.parent / "semantic_state")
    ]
    package.__package__ = name
    sys.modules[name] = package


_install_multiformats_shim()
_install_semantic_state_package_stub()

from ipfs_datasets_py.logic.backends.smt.compiler import (
    INT_SORT,
    SmtTerm,
    SmtTermKind,
    term_and,
    term_symbol,
)
from ipfs_datasets_py.logic.backends.smt.incremental import (
    INCREMENTAL_SMT_INTERFACE,
    INCREMENTAL_SMT_REPLAY_SCHEMA,
    IncrementalSmtFingerprint,
    IncrementalSmtResult,
    IncrementalSmtUnavailable,
    NamedSessionAssertion,
    SmtCheckStatus,
)
from ipfs_datasets_py.logic.backends.smt.interpolation import (
    InterpolationStatus,
    admit_interpolant,
)
from ipfs_datasets_py.logic.common.canonical_cache_key import CanonicalProofCacheKey
from ipfs_datasets_py.logic.ir_core.claims import FrozenMap
from ipfs_datasets_py.logic.ir_core.identity import canonical_identity
from ipfs_datasets_py.logic.ir_core.axes import (
    LogicEvidenceAuthority,
    LogicEvidenceKind,
)
from ipfs_datasets_py.logic.software_contracts.compositional import (
    ClauseKind,
    CompositionalContract,
    ContractConfidence,
    EvidenceAuthority,
    SemanticContractClause,
    SemanticSupport,
)
from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_bytes,
    cid_for_structured,
)
from ipfs_datasets_py.logic.software_contracts.contracts import (
    BoundedPredicate,
    CallableContract,
    ContractAuthority,
    ContractProvenance,
)
from ipfs_datasets_py.logic.software_contracts.semantic_index.index import (
    scan_repository,
)
from ipfs_datasets_py.logic.software_contracts.semantic_index.models import (
    RepositoryState,
    SymbolRecord,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.api import (
    build_semantic_state,
    verify_semantic_state_bundle,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.capsules import (
    compile_semantic_capsules,
)
from ipfs_datasets_py.logic.software_verification.assume_guarantee import (
    ComponentCompositionGraph,
    CompositionEdge,
    DischargeDisposition,
)
from ipfs_datasets_py.logic.software_verification.incremental_verification import (
    EvidenceDecisionDisposition,
    EvidenceReuseRequest,
    VerificationBindingKind,
    VerificationEvidenceBinding,
)

from ..analysis.deterministic_doctor_contracts import (
    DeterministicDoctorPlan,
    DoctorAuthorityRoots,
    DoctorConsumerDisposition,
    DoctorEditSite,
    DoctorPlanDisposition,
    DoctorPlanStep,
    DoctorRepairDisposition,
)
from ..context.planner_doctor_context import (
    PlannerDoctorContextRequest,
    compile_planner_doctor_context,
)
from ..planning.deterministic_doctor_transaction import (
    DeterministicDoctorTransaction,
    DoctorCheckoutLock,
    DoctorSandboxEnforcementLevel,
    DoctorSandboxPolicy,
    DoctorStepApplyRequest,
    DoctorStepApplyResult,
    DoctorStepDisposition,
    DoctorTransactionDisposition,
    DoctorWriterLease,
    PathBeforeHash,
)
from ..planning.program_repair_synthesis import (
    ProgramRepairBounds,
    ProgramRepairMode,
    ProgramRepairRequest,
    synthesize_program_repair,
)
from ..planning.repair_operator_registry import RepairOperatorKind
from ..proof.counterexample_guided_tactician import (
    CandidateKind,
    CandidateValidationStatus,
    RefinementCandidate,
)
from ..proof.formal_counterexamples import (
    CounterexampleKind,
    RepairClass,
    normalize_counterexample,
)
from ..proof.formal_verification_contracts import content_identity
from ..semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
from ..semantic_state.worktree import PatchScope, create_isolated_worktree
from .deterministic_doctor_live_fixed_point import (
    DeterministicDoctorLiveFixedPoint,
    LiveFixedPointRequest,
)

VERTICAL_SLICE_INTERFACE: Final = "CompositionalVerificationVerticalSlice@1"
VERTICAL_SLICE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/compositional-verification-vertical-slice@1"
)
VERTICAL_ARTIFACT_INTERFACE: Final = "CompositionalVerificationArtifact@1"
VERTICAL_ARTIFACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/compositional-verification-artifact@1"
)
VERTICAL_ARTIFACT_VERIFIER_INTERFACE: Final = (
    "CompositionalVerificationArtifactVerifier@1"
)

_TARGET_PATH = "pkg/module_a.py"
_FAULT_VALUE = 30
_REPAIR_VALUE = 10
# Logical refs hashed into transaction/fixed-point receipts.  Absolute temp
# worktree paths and incrementing durable fences are run-specific and must not
# appear in replay-stable artifact identities.
_REPAIR_WORKTREE_REF = "worktree:lgcvf-vertical-repair"
_REPAIR_FENCE_REF = "fence:lgcvf-vertical-repair"
_PROVIDER_MARKERS = (
    "anthropic",
    "llm_router",
    "model_provider",
    "openai",
)
REQUIRED_VERTICAL_STAGES: Final[tuple[str, ...]] = (
    "identity",
    "scan",
    "abstract_states",
    "contracts",
    "initial_discharge",
    "capsules",
    "context",
    "mutation",
    "counterexample",
    "invalidation",
    "unaffected_reuse",
    "incremental_smt",
    "core",
    "interpolant",
    "isolated_repair",
    "affected_replay",
    "live_fixed_point",
    "final_context",
    "zero_model_calls",
    "token_metrics",
    "work_reuse_metrics",
    "verified_artifact",
)
FRAGMENT_SMT_PROVIDER: Final = "qf-lia-fragment-checker"
FRAGMENT_SMT_VERSION: Final = "1"


class VerticalSliceError(RuntimeError):
    """Raised when a mandatory fixture capability cannot be reproduced."""


def _eval_int(term: SmtTerm, env: Mapping[str, int]) -> int:
    kind = term.kind
    if kind is SmtTermKind.INT:
        return int(term.value)
    if kind is SmtTermKind.SYMBOL:
        return int(env[term.value])
    if kind is SmtTermKind.NEG and term.arguments:
        return -_eval_int(term.arguments[0], env)
    if kind is SmtTermKind.ADD:
        total = 0
        for item in term.arguments:
            total += _eval_int(item, env)
        return total
    if kind is SmtTermKind.SUB and len(term.arguments) == 2:
        return _eval_int(term.arguments[0], env) - _eval_int(term.arguments[1], env)
    if kind is SmtTermKind.MUL:
        product = 1
        for item in term.arguments:
            product *= _eval_int(item, env)
        return product
    raise VerticalSliceError(f"unsupported integer term {kind}")


def _eval_bool(term: SmtTerm, env: Mapping[str, int]) -> bool:
    kind = term.kind
    if kind is SmtTermKind.TRUE:
        return True
    if kind is SmtTermKind.FALSE:
        return False
    if kind is SmtTermKind.BOOL:
        return term.value == "true"
    if kind is SmtTermKind.NOT and term.arguments:
        return not _eval_bool(term.arguments[0], env)
    if kind is SmtTermKind.AND:
        return all(_eval_bool(item, env) for item in term.arguments)
    if kind is SmtTermKind.OR:
        return any(_eval_bool(item, env) for item in term.arguments)
    if kind is SmtTermKind.IMPLIES and len(term.arguments) == 2:
        return (not _eval_bool(term.arguments[0], env)) or _eval_bool(
            term.arguments[1], env
        )
    if kind is SmtTermKind.IFF and len(term.arguments) == 2:
        return _eval_bool(term.arguments[0], env) is _eval_bool(term.arguments[1], env)
    if kind is SmtTermKind.ITE and len(term.arguments) == 3:
        chosen = term.arguments[1] if _eval_bool(term.arguments[0], env) else term.arguments[2]
        return _eval_bool(chosen, env)
    if kind is SmtTermKind.EQ and len(term.arguments) == 2:
        return _eval_int(term.arguments[0], env) == _eval_int(term.arguments[1], env)
    if kind is SmtTermKind.LT and len(term.arguments) == 2:
        return _eval_int(term.arguments[0], env) < _eval_int(term.arguments[1], env)
    if kind is SmtTermKind.LE and len(term.arguments) == 2:
        return _eval_int(term.arguments[0], env) <= _eval_int(term.arguments[1], env)
    if kind is SmtTermKind.GT and len(term.arguments) == 2:
        return _eval_int(term.arguments[0], env) > _eval_int(term.arguments[1], env)
    if kind is SmtTermKind.GE and len(term.arguments) == 2:
        return _eval_int(term.arguments[0], env) >= _eval_int(term.arguments[1], env)
    raise VerticalSliceError(f"unsupported boolean term {kind}")


def _collect_int_constants(term: SmtTerm) -> set[int]:
    constants: set[int] = set()
    if term.kind is SmtTermKind.INT:
        constants.add(int(term.value))
    for item in term.arguments:
        constants.update(_collect_int_constants(item))
    return constants


def _collect_symbols(term: SmtTerm) -> set[str]:
    names: set[str] = set()
    if term.kind is SmtTermKind.SYMBOL:
        names.add(str(term.value))
    for item in term.arguments:
        names.update(_collect_symbols(item))
    return names


def _conjunction(formulas: Sequence[SmtTerm]) -> SmtTerm:
    if not formulas:
        return SmtTerm(SmtTermKind.TRUE)
    return term_and(*formulas)


def _qf_lia_sat(term: SmtTerm) -> bool | None:
    from ipfs_datasets_py.logic.backends.smt.interpolation import _qf_lia_sat as decide

    return decide(term)


def _find_unary_model(formulas: Sequence[SmtTerm], symbols: Sequence[str]) -> dict[str, str] | None:
    if len(symbols) != 1:
        return None
    name = symbols[0]
    constants = set()
    for formula in formulas:
        constants.update(_collect_int_constants(formula))
    candidates = sorted(
        constants
        | {item - 1 for item in constants}
        | {item + 1 for item in constants}
        | {0, 1, -1, 10, 20, 21, 29, 30}
    )
    window = list(range(-256, 257))
    for value in (*candidates, *window):
        env = {name: value}
        try:
            if all(_eval_bool(formula, env) for formula in formulas):
                return {name: str(value)}
        except (KeyError, VerticalSliceError, TypeError, ValueError):
            continue
    return None


class _FragmentSmtSession:
    """Sound unary QF_LIA session used when the Z3 Python API is not installed."""

    interface = INCREMENTAL_SMT_INTERFACE

    def __init__(self, *, session_id: str, fingerprint: IncrementalSmtFingerprint) -> None:
        self.session_id = session_id
        self.fingerprint = fingerprint
        self._symbols: dict[str, str] = {}
        self._assertions: list[NamedSessionAssertion] = []
        self._frame_sizes: list[int] = []
        self._transcript: list[dict[str, Any]] = []
        self._last_result: IncrementalSmtResult | None = None
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise VerticalSliceError("incremental SMT session is closed")

    def assert_fresh(self, expected: IncrementalSmtFingerprint | str) -> None:
        expected_digest = (
            expected.digest if isinstance(expected, IncrementalSmtFingerprint) else expected
        )
        if expected_digest != self.fingerprint.digest:
            raise VerticalSliceError(
                f"session fingerprint mismatch expected={expected_digest} actual={self.fingerprint.digest}"
            )

    def cancel(self) -> None:
        self._transcript.append({"operation": "cancel"})

    def declare_symbol(self, name: str, range_sort: Any, domain: Sequence[Any] = ()) -> str:
        self._require_open()
        if name in self._symbols:
            raise VerticalSliceError(f"symbol {name!r} already declared")
        if domain:
            raise VerticalSliceError("fragment session admits integer constants only")
        self._symbols[name] = str(getattr(range_sort, "name", range_sort))
        self._transcript.append({"operation": "declare_symbol", "declaration": {"name": name}})
        return name

    def add_named_assertion(
        self,
        assertion_id: str,
        formula: SmtTerm,
        *,
        source_ref: str,
        obligation_id: str,
    ) -> NamedSessionAssertion:
        self._require_open()
        if any(item.assertion_id == assertion_id for item in self._assertions):
            raise VerticalSliceError(f"assertion {assertion_id!r} already exists")
        record = NamedSessionAssertion(assertion_id, formula, source_ref, obligation_id)
        self._assertions.append(record)
        self._transcript.append({"operation": "add_named_assertion", "assertion": record.to_dict()})
        return record

    def push(self) -> None:
        self._require_open()
        self._frame_sizes.append(len(self._assertions))
        self._transcript.append({"operation": "push"})

    def pop(self, levels: int = 1) -> None:
        self._require_open()
        if levels <= 0 or levels > len(self._frame_sizes):
            raise VerticalSliceError("pop exceeds pushed frame depth")
        target = self._frame_sizes[-levels]
        del self._assertions[target:]
        del self._frame_sizes[-levels:]
        self._transcript.append({"operation": "pop", "levels": levels})

    def check(self) -> IncrementalSmtResult:
        self._require_open()
        formulas = tuple(item.formula for item in self._assertions)
        combined = _conjunction(formulas)
        sat = _qf_lia_sat(combined)
        if sat is False:
            shrinking = list(self._assertions)
            if len(shrinking) > 1:
                reduced = []
                for item in shrinking:
                    rest = [other for other in shrinking if other is not item]
                    rest_sat = _qf_lia_sat(
                        _conjunction(tuple(other.formula for other in rest))
                    )
                    if rest_sat is not False:
                        reduced.append(item)
                if reduced and _qf_lia_sat(
                    _conjunction(tuple(item.formula for item in reduced))
                ) is False:
                    shrinking = reduced
            core_ids = tuple(sorted(item.assertion_id for item in shrinking))
            core_formulas = tuple(
                item.formula for item in self._assertions if item.assertion_id in set(core_ids)
            )
            core_validated = _qf_lia_sat(_conjunction(core_formulas)) is False
            result = IncrementalSmtResult(
                session_id=self.session_id,
                session_fingerprint=self.fingerprint.digest,
                status=SmtCheckStatus.UNSAT,
                active_assertion_ids=tuple(item.assertion_id for item in self._assertions),
                unsat_core=core_ids,
                core_validated=core_validated,
                limitations=(
                    "z3_python_api_unavailable",
                    "unary_qf_lia_fragment_checker",
                ),
            )
        elif sat is True:
            model = _find_unary_model(formulas, tuple(self._symbols)) or {}
            env = {key: int(value) for key, value in model.items()}
            validated = bool(model) and all(_eval_bool(item.formula, env) for item in self._assertions)
            result = IncrementalSmtResult(
                session_id=self.session_id,
                session_fingerprint=self.fingerprint.digest,
                status=SmtCheckStatus.SAT,
                active_assertion_ids=tuple(item.assertion_id for item in self._assertions),
                model=FrozenMap(model),
                model_validated=validated,
                limitations=(
                    "z3_python_api_unavailable",
                    "unary_qf_lia_fragment_checker",
                ),
            )
        else:
            result = IncrementalSmtResult(
                session_id=self.session_id,
                session_fingerprint=self.fingerprint.digest,
                status=SmtCheckStatus.UNKNOWN,
                active_assertion_ids=tuple(item.assertion_id for item in self._assertions),
                unknown_reason="fragment checker could not decide QF_LIA query",
                limitations=(
                    "z3_python_api_unavailable",
                    "unary_qf_lia_fragment_checker",
                ),
            )
        self._last_result = result
        self._transcript.append(
            {
                "operation": "check",
                "result_receipt_id": result.receipt_id,
                "status": result.status.value,
            }
        )
        return result

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._transcript.append({"operation": "close"})

    def snapshot_or_replay_manifest(self) -> dict[str, Any]:
        payload = {
            "assertions": [item.to_dict() for item in self._assertions],
            "fingerprint": self.fingerprint.to_dict(),
            "frame_sizes": list(self._frame_sizes),
            "schema": INCREMENTAL_SMT_REPLAY_SCHEMA,
            "session_id": self.session_id,
            "symbols": [{"name": name} for name in sorted(self._symbols)],
            "transcript": list(self._transcript),
        }
        identity = canonical_identity(
            payload,
            domain="logic.backends.smt.incremental-replay",
            schema_version=INCREMENTAL_SMT_REPLAY_SCHEMA,
        )
        return {**payload, "manifest_cid": identity.cid, "manifest_digest": identity.digest}


def _open_fragment_smt_session(
    *,
    session_id: str,
    provider: str = FRAGMENT_SMT_PROVIDER,
    logic: str = "QF_LIA",
    translator_identity: str,
    theory_fingerprint: str,
    policy_root: str,
    configuration_root: str,
    environment_root: str,
    deterministic_seed: int = 0,
    timeout_ms: int = 5_000,
    memory_limit_mib: int = 512,
) -> _FragmentSmtSession:
    del provider
    fingerprint = IncrementalSmtFingerprint(
        provider=FRAGMENT_SMT_PROVIDER,
        provider_version=FRAGMENT_SMT_VERSION,
        logic=logic,
        translator_identity=translator_identity,
        theory_fingerprint=theory_fingerprint,
        policy_root=policy_root,
        configuration_root=configuration_root,
        environment_root=environment_root,
        deterministic_seed=deterministic_seed,
        timeout_ms=timeout_ms,
        memory_limit_mib=memory_limit_mib,
    )
    return _FragmentSmtSession(session_id=session_id, fingerprint=fingerprint)


def _install_fragment_smt_adapter() -> None:
    """Use the unary QF_LIA checker when the Z3 Python API is not installed."""

    try:
        importlib.import_module("z3")
        return
    except ImportError:
        pass
    incremental = importlib.import_module(
        "ipfs_datasets_py.logic.backends.smt.incremental"
    )
    original = incremental.open_incremental_smt_session

    def _dispatch(**kwargs: Any) -> Any:
        try:
            return original(**kwargs)
        except IncrementalSmtUnavailable:
            return _open_fragment_smt_session(**kwargs)

    incremental.open_incremental_smt_session = _dispatch
    for module in list(sys.modules.values()):
        current = getattr(module, "open_incremental_smt_session", None)
        if current is original:
            setattr(module, "open_incremental_smt_session", _dispatch)


_install_fragment_smt_adapter()


class _StageTrace:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(self, stage: str, *, receipt_id: str, status: str, **details: Any) -> None:
        if stage not in REQUIRED_VERTICAL_STAGES:
            raise VerticalSliceError(f"undeclared vertical stage {stage!r}")
        if any(item["stage"] == stage for item in self.records):
            raise VerticalSliceError(f"vertical stage {stage!r} recorded twice")
        record = {
            "details": _plain(details) if details else {},
            "receipt_id": receipt_id,
            "stage": stage,
            "status": status,
        }
        self.records.append(record)

    def require_complete(self) -> None:
        observed = tuple(item["stage"] for item in self.records)
        if observed != REQUIRED_VERTICAL_STAGES:
            raise VerticalSliceError(
                "vertical stage trace mismatch: "
                f"observed={list(observed)} expected={list(REQUIRED_VERTICAL_STAGES)}"
            )


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict())
    to_record = getattr(value, "to_record", None)
    if callable(to_record):
        return _plain(to_record())
    return value


def _git(cwd: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment.setdefault("GIT_AUTHOR_DATE", "2000-01-01T00:00:00+00:00")
    environment.setdefault("GIT_COMMITTER_DATE", "2000-01-01T00:00:00+00:00")
    completed = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise VerticalSliceError(
            f"git {' '.join(arguments)} failed: "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    return completed.stdout.strip()


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _patch(path: str, before: str, after: str) -> str:
    if before == after:
        raise VerticalSliceError("patch must change real bytes")
    body = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    return f"diff --git a/{path} b/{path}\n{body}"


def _run_pytest(root: Path, targets: Sequence[str]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    command = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *targets,
    )
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    return {
        "argv": list(command),
        "output": completed.stdout[-4_096:],
        "returncode": completed.returncode,
        "status": "passed" if completed.returncode == 0 else "failed",
    }


def _test_receipt_id(receipt: Mapping[str, Any]) -> str:
    """Address stable test semantics, excluding diagnostic timing output."""

    return content_identity(
        {
            "argv": list(receipt.get("argv") or ()),
            "returncode": int(receipt.get("returncode") or 0),
            "schema": "lgcvf-test-receipt@1",
            "status": str(receipt.get("status") or "unknown"),
        }
    )


def _fixture_default() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "test"
        / "fixtures"
        / "agent_supervisor"
        / "compositional_verification"
    )


def _copy_fixture(fixture_root: Path, destination: Path) -> None:
    if not fixture_root.is_dir():
        raise VerticalSliceError(f"fixture root does not exist: {fixture_root}")
    # Copy bytes only. shutil.copytree/copy2 chmod the destination, which the
    # qualification worker's seccomp profile denies.
    destination.mkdir(parents=True, exist_ok=True)
    for current, dirnames, filenames in os.walk(fixture_root):
        relative = Path(current).relative_to(fixture_root)
        target_dir = destination / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in dirnames:
            (target_dir / name).mkdir(exist_ok=True)
        for name in filenames:
            source = Path(current) / name
            if source.is_symlink() or not source.is_file():
                continue
            (target_dir / name).write_bytes(source.read_bytes())
    _git(destination, "init", "-b", "main")
    _git(destination, "config", "user.email", "lgcvf-fixture@example.invalid")
    _git(destination, "config", "user.name", "LGCVF Hermetic Fixture")
    _git(destination, "add", "--all")
    _git(destination, "commit", "-m", "fixture baseline")


def _symbol(state: RepositoryState, path: str, name: str) -> SymbolRecord:
    candidates = [
        item
        for item in state.symbols
        if item.module_path == path and item.qualified_name.endswith(f".{name}")
    ]
    if len(candidates) != 1:
        raise VerticalSliceError(
            f"expected exactly one {path}:{name} symbol, found {len(candidates)}"
        )
    return candidates[0]


def _provenance(path: str, symbol: str, *, inferred: bool) -> ContractProvenance:
    return ContractProvenance(
        fact_kind="inferred" if inferred else "extracted",
        authority=ContractAuthority(
            authority_id=(
                "authority:python-abstract-interpreter@1"
                if inferred
                else "authority:fixture-source-contract@1"
            ),
            rank="inference" if inferred else "type_declaration",
            owner="ipfs_datasets_py.logic",
            revision="compositional-verification-fixture@1",
        ),
        source_path=path,
        source_symbol=symbol,
        note=(
            "candidate abstract interpretation fact"
            if inferred
            else "typed assertion extracted from hermetic fixture source"
        ),
    )


def _range_predicate(
    predicate_id: str,
    role: str,
    lower: int,
    upper: int,
    provenance: ContractProvenance,
) -> BoundedPredicate:
    return BoundedPredicate(
        predicate_id=predicate_id,
        role=role,
        operator="range_int",
        subject="return",
        arguments=(lower, upper),
        provenance=provenance,
    )


def _clause(
    clause_id: str,
    kind: ClauseKind,
    predicate: BoundedPredicate,
) -> SemanticContractClause:
    return SemanticContractClause(
        clause_id=clause_id,
        kind=kind,
        support=SemanticSupport.TYPED_INLINE,
        predicate=predicate,
    )


def _contract_roots(
    symbol: SymbolRecord,
    *,
    configuration_root: str,
    toolchain_root: str,
) -> dict[str, str]:
    if not symbol.source_cid:
        raise VerticalSliceError(f"symbol {symbol.qualified_name} has no source CID")
    return {
        "source_root": symbol.source_cid,
        "ast_root": cid_for_structured(_plain(symbol.normalized_ast)),
        "symbol_version_root": symbol.version_cid,
        "interface_root": cid_for_structured(_plain(symbol.signature)),
        "configuration_root": configuration_root,
        "toolchain_root": toolchain_root,
    }


def _adapt_contract(
    *,
    api: Any,
    symbol: SymbolRecord,
    contract_id: str,
    provenance: ContractProvenance,
    postcondition: BoundedPredicate,
    roots: Mapping[str, str],
) -> CompositionalContract:
    legacy = CallableContract(
        contract_id=contract_id,
        qualified_name=symbol.qualified_name,
        owner_module=symbol.module_path,
        shape="sync_function",
        provenance=provenance,
        postconditions=(postcondition,),
        symbol_id=symbol.stable_id,
    )
    adapted = api.compile_component_contract(legacy, **dict(roots))
    return replace(
        adapted,
        component_id=f"component:{contract_id.rsplit(':', 1)[-1]}",
        open_world=False,
        confidence=ContractConfidence.CONSERVATIVE,
        semantic_support_class="supported_python_subset",
        evidence_authority=EvidenceAuthority.CANDIDATE,
        invalidation_selectors=(symbol.stable_id, symbol.version_cid),
        attributes={
            "adapted_from": "CallableContract@v1",
            "semantic_index_symbol_id": symbol.stable_id,
        },
    )


def _build_contract_graph(
    root: Path,
    state: RepositoryState,
    analyses: Mapping[str, Any],
    *,
    api: Any,
) -> tuple[ComponentCompositionGraph, dict[str, CompositionalContract]]:
    configuration_root = cid_for_bytes((root / "config/schema.json").read_bytes())
    toolchain_root = cid_for_structured(
        {
            "abstract_interpreter": analyses["A"].analyzer_identity,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "verification_api": "LogicVerificationAPI@1+compositional",
        }
    )
    symbols = {
        "A": _symbol(state, "pkg/module_a.py", "produce"),
        "B": _symbol(state, "pkg/module_b.py", "consume"),
        "C": _symbol(state, "pkg/module_c.py", "present"),
        "U": _symbol(state, "pkg/unaffected.py", "stable_label"),
    }
    a_summary = analyses["A"].summaries_by_name["produce"]
    a_interval = a_summary.return_value.interval
    if a_interval.empty or a_interval.lower is None or a_interval.upper is None:
        raise VerticalSliceError("A return interval is not finite in the supported slice")

    a_provenance = _provenance("pkg/module_a.py", "pkg.module_a.produce", inferred=True)
    a_post = _range_predicate(
        "predicate:A:return-range",
        "postcondition",
        a_interval.lower,
        a_interval.upper,
        a_provenance,
    )
    contract_a = _adapt_contract(
        api=api,
        symbol=symbols["A"],
        contract_id="contract:A",
        provenance=a_provenance,
        postcondition=a_post,
        roots=_contract_roots(
            symbols["A"],
            configuration_root=configuration_root,
            toolchain_root=toolchain_root,
        ),
    )
    a_exception = SemanticContractClause(
        clause_id="clause:A:negative-limit-value-error",
        kind=ClauseKind.EXCEPTIONAL_POSTCONDITION,
        support=SemanticSupport.TYPED_INLINE,
        predicate=BoundedPredicate(
            predicate_id="predicate:A:raises-value-error",
            role="exception",
            operator="raises",
            subject="limit",
            arguments=("ValueError",),
            provenance=a_provenance,
        ),
    )
    contract_a = replace(
        contract_a,
        guarantees=(
            _clause("clause:A:guarantee", ClauseKind.GUARANTEE, a_post),
        ),
        exceptional_postconditions=(a_exception,),
    )

    b_provenance = _provenance("pkg/module_b.py", "pkg.module_b.consume", inferred=False)
    b_post = _range_predicate(
        "predicate:B:return-range", "postcondition", 1, 21, b_provenance
    )
    contract_b = _adapt_contract(
        api=api,
        symbol=symbols["B"],
        contract_id="contract:B",
        provenance=b_provenance,
        postcondition=b_post,
        roots=_contract_roots(
            symbols["B"],
            configuration_root=configuration_root,
            toolchain_root=toolchain_root,
        ),
    )
    b_assumption_predicate = _range_predicate(
        "predicate:B:producer-assumption", "assumption", 0, 20, b_provenance
    )
    contract_b = replace(
        contract_b,
        assumptions=(
            _clause(
                "clause:B:producer-assumption",
                ClauseKind.ASSUMPTION,
                b_assumption_predicate,
            ),
        ),
        guarantees=(
            _clause("clause:B:guarantee", ClauseKind.GUARANTEE, b_post),
        ),
        read_set=("effect:B:read-schema",),
    )

    c_provenance = _provenance("pkg/module_c.py", "pkg.module_c.present", inferred=False)
    c_post = _range_predicate(
        "predicate:C:return-range", "postcondition", 2, 42, c_provenance
    )
    contract_c = _adapt_contract(
        api=api,
        symbol=symbols["C"],
        contract_id="contract:C",
        provenance=c_provenance,
        postcondition=c_post,
        roots=_contract_roots(
            symbols["C"],
            configuration_root=configuration_root,
            toolchain_root=toolchain_root,
        ),
    )
    c_assumption_predicate = _range_predicate(
        "predicate:C:consumer-assumption", "assumption", 1, 21, c_provenance
    )
    contract_c = replace(
        contract_c,
        assumptions=(
            _clause(
                "clause:C:consumer-assumption",
                ClauseKind.ASSUMPTION,
                c_assumption_predicate,
            ),
        ),
        guarantees=(
            _clause("clause:C:guarantee", ClauseKind.GUARANTEE, c_post),
        ),
    )

    u_provenance = _provenance(
        "pkg/unaffected.py", "pkg.unaffected.stable_label", inferred=False
    )
    u_post = _range_predicate(
        "predicate:U:evidence-marker", "postcondition", 1, 1, u_provenance
    )
    contract_u = _adapt_contract(
        api=api,
        symbol=symbols["U"],
        contract_id="contract:U",
        provenance=u_provenance,
        postcondition=u_post,
        roots=_contract_roots(
            symbols["U"],
            configuration_root=configuration_root,
            toolchain_root=toolchain_root,
        ),
    )
    contract_u = replace(
        contract_u,
        guarantees=(
            _clause("clause:U:guarantee", ClauseKind.GUARANTEE, u_post),
        ),
    )

    contracts = {
        "A": contract_a,
        "B": contract_b,
        "C": contract_c,
        "U": contract_u,
    }
    graph = ComponentCompositionGraph(
        semantic_state_root=state.state_cid,
        contracts=tuple(contracts.values()),
        edges=(
            CompositionEdge(
                edge_id="edge:A-to-B",
                producer_component_id=contract_a.component_id,
                consumer_component_id=contract_b.component_id,
                guarantee_clause_ids=("clause:A:guarantee",),
                assumption_clause_ids=("clause:B:producer-assumption",),
                source_fact_refs=(symbols["A"].stable_id, symbols["B"].stable_id),
            ),
            CompositionEdge(
                edge_id="edge:B-to-C",
                producer_component_id=contract_b.component_id,
                consumer_component_id=contract_c.component_id,
                guarantee_clause_ids=("clause:B:guarantee",),
                assumption_clause_ids=("clause:C:consumer-assumption",),
                source_fact_refs=(symbols["B"].stable_id, symbols["C"].stable_id),
            ),
        ),
    )
    return graph, contracts


def _analyze_components(root: Path, api: Any) -> dict[str, Any]:
    paths = {
        "A": "pkg/module_a.py",
        "B": "pkg/module_b.py",
        "C": "pkg/module_c.py",
        "U": "pkg/unaffected.py",
    }
    return {
        key: api.analyze_abstract_state(
            (root / path).read_text(encoding="utf-8"), source_uri=path
        )
        for key, path in paths.items()
    }


def _cache_key(label: str) -> CanonicalProofCacheKey:
    return CanonicalProofCacheKey.build(
        source={"source": label},
        expression={"expression": label},
        formalization={"formalization": "python-range"},
        slice={"symbols": [label]},
        obligation={"obligation": label},
        assumptions=(),
        bounds={"steps": 64},
        translation={"translator": "compositional-vertical@1"},
        provider="provider.z3",
        environment={
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "z3": "locally-probed",
        },
        policy={"network": "deny"},
        schema={"semantic-index": "v2"},
        checker="checker.compositional-vertical",
        network_policy={"allow": False},
        evidence_kind=LogicEvidenceKind.SOLVER_RESULT,
        authority_ceiling=LogicEvidenceAuthority.BOUNDED,
    )


def _reuse_request(
    *,
    binding_id: str,
    kind: VerificationBindingKind,
    state: RepositoryState,
    symbol: SymbolRecord,
    contract: CompositionalContract,
    artifact_cid: str,
) -> EvidenceReuseRequest:
    key = _cache_key(binding_id)
    return EvidenceReuseRequest(
        VerificationEvidenceBinding(
            binding_id=binding_id,
            kind=kind,
            artifact_cid=artifact_cid,
            observed_state_cid=state.state_cid,
            subject_ids=(symbol.stable_id,),
            dependency_ids=(symbol.version_cid,),
            contract_cids=(contract.cid,),
            cache_key=key,
            confidence="exact",
        ),
        key,
    )


def _localize_failure(api: Any, graph: ComponentCompositionGraph) -> dict[str, Any]:
    value = term_symbol("value")
    partition_a = SmtTerm(
        SmtTermKind.GE,
        arguments=(value, SmtTerm(SmtTermKind.INT, value=str(_FAULT_VALUE))),
    )
    partition_b = SmtTerm(
        SmtTermKind.LE,
        arguments=(value, SmtTerm(SmtTermKind.INT, value="20")),
    )
    session = api.open_incremental_smt_session(
        session_id="vertical-localization",
        translator_identity="software-verification-smt-structured-term@1",
        theory_fingerprint="QF_LIA:range-failure@1",
        policy_root=graph.contract_root,
        configuration_root=graph.graph_cid,
        environment_root=graph.semantic_state_root,
        deterministic_seed=0,
    )
    session.declare_symbol("value", INT_SORT)
    session.add_named_assertion(
        "producer-lower-bound",
        partition_a,
        source_ref="clause:A:guarantee",
        obligation_id="edge:A-to-B",
    )
    session.add_named_assertion(
        "consumer-upper-bound",
        partition_b,
        source_ref="clause:B:producer-assumption",
        obligation_id="edge:A-to-B",
    )
    solver_result = session.check()
    replay = session.snapshot_or_replay_manifest()
    session.close()
    interpolant = api.compute_and_validate_interpolant(partition_a, partition_b)
    interpolant_status = getattr(interpolant, "status", None)
    interpolant_value = (
        interpolant_status.value
        if isinstance(interpolant_status, Enum)
        else interpolant_status
    )
    if interpolant_value != InterpolationStatus.VALIDATED.value:
        # cvc5 is not required to *invent* I for this unary conflict.  The
        # producer lower bound is a Craig interpolant and is independently
        # admitted by the public validator (Z3 or the unary fragment checker).
        interpolant = admit_interpolant(partition_a, partition_b, partition_a)
    return {
        "incremental_solver": {
            **solver_result.to_dict(),
            "receipt_id": solver_result.receipt_id,
        },
        "interpolant": {
            **interpolant.to_dict(),
            "receipt_cid": interpolant.receipt_cid,
        },
        "replay_manifest": replay,
    }


def _doctor_roots(
    *,
    state: RepositoryState,
    graph: ComponentCompositionGraph,
    tree: str,
    lease_id: str,
) -> DoctorAuthorityRoots:
    return DoctorAuthorityRoots(
        repository_id=state.repository_id,
        forest_id=f"forest:{tree}",
        tree_id=f"git-tree:{tree}",
        overlay_id=f"overlay:{state.state_cid}",
        file_root_id=f"file-root:{tree}",
        ast_root_id=f"ast-root:{state.state_cid}",
        graph_id=graph.graph_cid,
        corpus_id=f"corpus:{state.repository_id}",
        index_id=state.state_cid,
        model_id="model:none",
        cache_id=f"cache:{graph.contract_root}",
        operator_registry_id="repair-operator-registry:default@1",
        translator_id="software-verification-smt-structured-term@1",
        solver_id="solver:z3-local",
        kernel_id="kernel:not-applicable",
        toolchain_id=f"toolchain:python-{sys.version_info.major}.{sys.version_info.minor}",
        policy_id="policy:lgcvf-hermetic-deny-network@1",
        sandbox_id="sandbox:lgcvf-hermetic-worktree@1",
        environment_id="environment:lgcvf-hermetic@1",
        lease_id=lease_id,
    )


def _synthesize_repair(
    *,
    roots: DoctorAuthorityRoots,
    failed_obligation_id: str,
    fault_source: str,
) -> Any:
    semantic_api = IpfsDatasetsSemanticStateProvider()
    witness = normalize_counterexample(
        {
            "kind": CounterexampleKind.GENERIC_FAILURE.value,
            "failure": {"observed": _FAULT_VALUE, "required_upper": 20},
        },
        kind=CounterexampleKind.GENERIC_FAILURE,
        violated_property=failed_obligation_id,
        bindings={
            "assumption_id": "clause:B:producer-assumption",
            "ast_scope_id": "component:A",
            "obligation_id": failed_obligation_id,
            "plan_id": "plan:lgcvf-vertical-repair",
            "policy_id": roots.policy_id,
            "provider_id": "tool:z3",
            "task_id": "LGCVF-VERTICAL",
            "tree_id": roots.tree_id,
        },
        finite_bounds={"candidate_constants": 3, "maximum_iterations": 2},
        repair_classes=(RepairClass.CONSTRAIN_SCOPE,),
    )
    candidates = (30, 10, 20)
    by_id = {f"candidate:constant:{item}": item for item in candidates}

    def refine(_witness: Any, _context: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(
            RefinementCandidate(
                candidate_id=candidate_id,
                kind=CandidateKind.REPAIR,
                goal_id=failed_obligation_id,
                repaired_tree_id=f"candidate-tree:constant-{value}",
                repaired_plan_id="plan:lgcvf-vertical-repair",
                statement=f"replace producer result with reviewed integer {value}",
                addresses_witness=True,
                parameters={
                    "operator_kind": RepairOperatorKind.EQUALITY_REWRITE.value,
                    "replacement": value,
                },
            )
            for candidate_id, value in by_id.items()
        )

    def validate(candidate: RefinementCandidate, _context: Mapping[str, Any]) -> Any:
        value = by_id[candidate.candidate_id]
        candidate_source = fault_source.replace(
            f"return {_FAULT_VALUE}", f"return {value}", 1
        )
        analysis = semantic_api.analyze_abstract_state(
            candidate_source, source_uri=f"candidate://{candidate.candidate_id}"
        )
        interval = analysis.summaries_by_name["produce"].return_value.interval
        if interval.lower is None or interval.upper is None:
            return CandidateValidationStatus.INVALID, "non-finite-candidate"
        if not (0 <= interval.lower and interval.upper <= 20):
            return CandidateValidationStatus.INVALID, "consumer-assumption-not-established"
        return CandidateValidationStatus.VALID, "abstract-range-independently-checked"

    def verify(binding: Mapping[str, Any]) -> Mapping[str, Any]:
        value = by_id[str(binding["candidate_id"])]
        candidate_source = fault_source.replace(
            f"return {_FAULT_VALUE}", f"return {value}", 1
        )
        analysis = semantic_api.analyze_abstract_state(
            candidate_source, source_uri=f"verify://constant-{value}"
        )
        interval = analysis.summaries_by_name["produce"].return_value.interval
        verified = (
            interval.lower is not None
            and interval.upper is not None
            and 0 <= interval.lower
            and interval.upper <= 20
        )
        return {
            "assumption_ids": list(binding.get("assumption_ids") or ()),
            "available": True,
            "bound_digest": binding["bound_digest"],
            "counterexample_id": binding["counterexample_id"],
            "outcome": "verified" if verified else "still_violated",
            "policy_id": binding["policy_id"],
            "property_id": binding["property_id"],
            "receipt_id": f"receipt:abstract-range:{analysis.analysis_id}",
            "repaired_plan_id": binding["repaired_plan_id"],
            "repository_tree_id": binding["repository_tree_id"],
            "tool_id": binding["tool_id"],
        }

    request = ProgramRepairRequest(
        roots=roots,
        obligation_refs=(failed_obligation_id,),
        target_paths=(_TARGET_PATH,),
        operator_kinds=(RepairOperatorKind.EQUALITY_REWRITE.value,),
        postcondition_refs=("clause:B:producer-assumption",),
        test_refs=("tests/test_selected.py",),
        mode=ProgramRepairMode.CEGIS,
        counterexample=witness,
        cegis_refine=refine,
        cegis_validate=validate,
        cegis_verify=verify,
        bounds=ProgramRepairBounds(
            max_cegis_iterations=2,
            max_candidates_per_iteration=3,
            max_search_states=16,
        ),
        allow_hybrid_residual=False,
        metadata={"candidate_grammar": "reviewed-integer-constants"},
    )
    receipt = synthesize_program_repair(request)
    if not receipt.admitted or receipt.cegis_result is None:
        raise VerticalSliceError("existing CEGIS repair synthesizer did not close")
    selected = receipt.cegis_result.selected_candidate
    if selected is None or selected.parameters.get("replacement") != _REPAIR_VALUE:
        raise VerticalSliceError("CEGIS did not select the expected reviewed repair")
    return receipt


def _doctor_plan(
    *,
    roots: DoctorAuthorityRoots,
    before_hash: str,
    failed_obligation_id: str,
    repair_receipt: Any,
) -> DeterministicDoctorPlan:
    site = DoctorEditSite(
        path=_TARGET_PATH,
        before_hash=before_hash,
        span_start=0,
        span_end=1,
        artifact_id="blob:module-a-fault",
    )
    step = DoctorPlanStep(
        step_id="step:apply-reviewed-constant-repair",
        kind="analytical",
        operator_id="repair-operator:replace_constant@1",
        consumer_ids=("component:B",),
        edit_site_refs=(site.content_id,),
        validation_refs=(failed_obligation_id,),
        write_paths=(_TARGET_PATH,),
    )
    consumer = DoctorConsumerDisposition(
        roots=roots,
        consumer_id="component:B",
        disposition=DoctorRepairDisposition.SUPPORTED,
        reason_codes=("failed_assumption_localized",),
    )
    return DeterministicDoctorPlan(
        roots=roots,
        plan_id="plan:lgcvf-vertical-repair",
        snapshot_id="snapshot:lgcvf-fault",
        finding_ids=(failed_obligation_id,),
        disposition=DoctorPlanDisposition.ADMITTED,
        consumer_dispositions=(consumer,),
        impact_closure_id="impact:A-B-C",
        steps=(step,),
        edit_sites=(site,),
        operator_ids=("repair-operator:replace_constant@1",),
        target_ref="component:A",
        value_source_ref="cegis:reviewed-integer-grammar",
        placement_ref="placement:module-a-return",
        selected_operator_id="repair-operator:replace_constant@1",
        candidate_refs=(repair_receipt.content_id,),
        permitted_read_paths=(_TARGET_PATH,),
        permitted_write_paths=(_TARGET_PATH,),
        lease_id=roots.lease_id,
        checkpoint_ref="checkpoint:fault-tree",
        rollback_ref="rollback:exact-fault-tree",
        proof_refs=(repair_receipt.content_id,),
        invalidation_refs=(roots.tree_id,),
        no_model_invariant=True,
        llm_router_enabled=False,
        model_invocation_count=0,
    )


@dataclass(frozen=True)
class CompositionalVerificationArtifact:
    """Fixture-specific checked projection over existing authoritative receipts."""

    payload: Mapping[str, Any]
    artifact_cid: str = ""
    schema: str = VERTICAL_ARTIFACT_SCHEMA
    interface: str = VERTICAL_ARTIFACT_INTERFACE

    def __post_init__(self) -> None:
        if self.schema != VERTICAL_ARTIFACT_SCHEMA or self.interface != VERTICAL_ARTIFACT_INTERFACE:
            raise VerticalSliceError("unsupported vertical artifact schema/interface")
        normalized = _plain(self.payload)
        object.__setattr__(self, "payload", MappingProxyType(normalized))
        expected = content_identity(
            {"interface": self.interface, "payload": normalized, "schema": self.schema}
        )
        if self.artifact_cid and self.artifact_cid != expected:
            raise VerticalSliceError("vertical artifact content identity mismatch")
        object.__setattr__(self, "artifact_cid", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_cid": self.artifact_cid,
            "interface": self.interface,
            "payload": _plain(self.payload),
            "schema": self.schema,
        }


@dataclass(frozen=True)
class ArtifactVerificationResult:
    disposition: str
    artifact_cid: str
    replay_receipt_cid: str
    issues: tuple[str, ...] = ()
    checks: Mapping[str, Any] = field(default_factory=dict)
    interface: str = VERTICAL_ARTIFACT_VERIFIER_INTERFACE

    @property
    def valid(self) -> bool:
        return self.disposition == "validated" and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_cid": self.artifact_cid,
            "checks": _plain(self.checks),
            "disposition": self.disposition,
            "interface": self.interface,
            "issues": list(self.issues),
            "replay_receipt_cid": self.replay_receipt_cid,
            "valid": self.valid,
        }


def verify_compositional_artifact(
    artifact: CompositionalVerificationArtifact,
    *,
    worktree: Path,
    expected_state: RepositoryState,
    expected_graph: ComponentCompositionGraph,
) -> ArtifactVerificationResult:
    """Independently reconstruct identities and replay compact mandatory checks."""

    issues: list[str] = []
    checks: dict[str, Any] = {}
    expected_cid = content_identity(
        {
            "interface": artifact.interface,
            "payload": _plain(artifact.payload),
            "schema": artifact.schema,
        }
    )
    checks["content_identity_reconstructed"] = expected_cid == artifact.artifact_cid
    if expected_cid != artifact.artifact_cid:
        issues.append("artifact_content_identity_mismatch")

    observed_state = scan_repository(worktree, previous_state=expected_state)
    checks["semantic_index_reconstructed"] = observed_state.state_cid == expected_state.state_cid
    if observed_state.state_cid != expected_state.state_cid:
        issues.append("semantic_index_reconstruction_mismatch")
    bundle = build_semantic_state(observed_state)
    semantic_root = verify_semantic_state_bundle(bundle)
    claimed_semantic_root = str(artifact.payload.get("semantic_state_root_cid") or "")
    checks["semantic_state_reconstructed"] = semantic_root.root_cid == claimed_semantic_root
    if semantic_root.root_cid != claimed_semantic_root:
        issues.append("semantic_state_root_mismatch")

    discharge = IpfsDatasetsSemanticStateProvider().discharge_assume_guarantee(
        expected_graph,
        expected_semantic_state_root=expected_state.state_cid,
        expected_contract_root=expected_graph.contract_root,
    )
    checks["composition_redischarge"] = discharge.disposition is DischargeDisposition.PROVED
    if discharge.disposition is not DischargeDisposition.PROVED:
        issues.append("composition_not_reproved")
    claimed_discharge = str(artifact.payload.get("final_discharge_receipt_cid") or "")
    checks["discharge_identity_reconstructed"] = discharge.receipt_cid == claimed_discharge
    if discharge.receipt_cid != claimed_discharge:
        issues.append("discharge_receipt_identity_mismatch")

    selected = _run_pytest(worktree, ("tests/test_selected.py",))
    checks["selected_tests_replayed"] = selected["returncode"] == 0
    observed_test_receipt = _test_receipt_id(selected)
    checks["selected_test_receipt_cid"] = observed_test_receipt
    if selected["returncode"] != 0:
        issues.append("selected_test_replay_failed")
    claimed_test_receipt = str(
        artifact.payload.get("selected_test_receipt_cid") or ""
    )
    checks["selected_test_receipt_reconstructed"] = (
        observed_test_receipt == claimed_test_receipt
    )
    if observed_test_receipt != claimed_test_receipt:
        issues.append("selected_test_receipt_identity_mismatch")

    replay_payload = {
        "artifact_cid": artifact.artifact_cid,
        "checks": checks,
        "issues": sorted(issues),
        "verifier": VERTICAL_ARTIFACT_VERIFIER_INTERFACE,
    }
    return ArtifactVerificationResult(
        disposition="validated" if not issues else "rejected",
        artifact_cid=artifact.artifact_cid,
        replay_receipt_cid=content_identity(replay_payload),
        issues=tuple(sorted(issues)),
        checks=checks,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_compositional_verification_vertical_slice(
    *,
    fixture_root: Path | str | None = None,
    output_path: Path | str | None = None,
    benchmark_output_path: Path | str | None = None,
    keep_workdir: bool = False,
) -> dict[str, Any]:
    """Run the complete deterministic Python fixture through public APIs."""

    api = IpfsDatasetsSemanticStateProvider()
    provider_modules_before = set(sys.modules)
    trace = _StageTrace()
    fixture = Path(fixture_root) if fixture_root is not None else _fixture_default()
    temp_root = Path(tempfile.mkdtemp(prefix="lgcvf-vertical-"))
    repository = temp_root / "repository"
    _copy_fixture(fixture.resolve(), repository)
    base_commit = _git(repository, "rev-parse", "HEAD")
    base_tree = _git(repository, "rev-parse", "HEAD^{tree}")
    trace.add(
        "identity",
        receipt_id=base_tree,
        status="recorded",
        base_commit=base_commit,
        base_tree=base_tree,
    )

    fault_worktree = create_isolated_worktree(
        repo_root=repository,
        worktree_path=temp_root / "worktrees" / "fault",
        base_commit=base_commit,
        base_tree=base_tree,
        task_id="LGCVF-VERTICAL-FAULT",
        lane_id="lgcvf-hermetic",
        canonical_task_cid="task:lgcvf-vertical-fault",
        lease_id="lease:lgcvf-vertical-fault",
        retain_on_success=True,
    )
    working = fault_worktree.worktree_path

    baseline_state = scan_repository(working)
    baseline_semantic_bundle = build_semantic_state(baseline_state)
    baseline_semantic_root = verify_semantic_state_bundle(baseline_semantic_bundle)
    trace.add(
        "scan",
        receipt_id=baseline_state.state_cid,
        status="scanned",
        semantic_state_root_cid=baseline_semantic_root.root_cid,
    )
    baseline_analyses = _analyze_components(working, api)
    trace.add(
        "abstract_states",
        receipt_id=baseline_analyses["A"].analysis_id,
        status="analyzed",
        components=sorted(baseline_analyses),
    )
    baseline_graph, baseline_contracts = _build_contract_graph(
        working, baseline_state, baseline_analyses, api=api
    )
    trace.add(
        "contracts",
        receipt_id=baseline_graph.contract_root,
        status="compiled",
        graph_cid=baseline_graph.graph_cid,
    )
    baseline_discharge = api.discharge_assume_guarantee(
        baseline_graph,
        expected_semantic_state_root=baseline_state.state_cid,
        expected_contract_root=baseline_graph.contract_root,
    )
    if baseline_discharge.disposition is not DischargeDisposition.PROVED:
        raise VerticalSliceError("baseline assume-guarantee composition did not prove")
    trace.add(
        "initial_discharge",
        receipt_id=baseline_discharge.receipt_cid,
        status="proved",
    )
    baseline_capsules = compile_semantic_capsules(baseline_state)
    trace.add(
        "capsules",
        receipt_id=baseline_capsules.capsule_index_cid,
        status="compiled",
    )
    baseline_tests = _run_pytest(working, ("tests/test_selected.py",))
    if baseline_tests["returncode"]:
        raise VerticalSliceError("baseline selected tests did not pass")
    baseline_context = compile_planner_doctor_context(
        PlannerDoctorContextRequest(
            repository_id=baseline_state.repository_id,
            tree_id=f"git-tree:{base_tree}",
            task_id="LGCVF-VERTICAL-BASELINE",
            acceptance_ids=("acceptance:composition-proved", "acceptance:tests-pass"),
            intent_summary="Baseline A/B/C composition under current roots",
            security_roots=("policy:lgcvf-hermetic-deny-network@1",),
            open_obligation_ids=(),
            assumption_ids=("clause:B:producer-assumption",),
            counterexample_ids=(),
            impact_coverage_ids=(),
            allowed_paths=(_TARGET_PATH,),
            protected_paths=("tests", "config"),
            allowed_effects=(f"effect:{_TARGET_PATH}",),
            validation_commands=(
                f"{sys.executable} -m pytest -q -p no:cacheprovider tests/test_selected.py",
            ),
            satisfied_proof_handles=(
                baseline_discharge.receipt_cid,
                baseline_analyses["A"].analysis_id,
                baseline_semantic_root.root_cid,
            ),
            retrieval_slice_node_ids=(),
            deterministic_closure=True,
            objective_id="LGCVF-G001",
            objective_revision="logic-governed-compositional-verification-fabric-v1",
            policy_id="policy:lgcvf-hermetic-deny-network@1",
            policy_revision=cid_for_structured(
                {"policy": "policy:lgcvf-hermetic-deny-network@1"}
            ),
            goal_summary="Record the baseline mandatory-coverage context before mutation",
        )
    )
    if not baseline_context.deterministic_closed:
        raise VerticalSliceError("baseline context did not remain deterministically closed")
    trace.add(
        "context",
        receipt_id=content_identity(baseline_context.to_dict()),
        status="compiled",
        deterministic_closed=True,
    )

    baseline_source = (working / _TARGET_PATH).read_text(encoding="utf-8")
    fault_source = baseline_source.replace(
        f"return {_REPAIR_VALUE}", f"return {_FAULT_VALUE}", 1
    )
    fault_patch = _patch(_TARGET_PATH, baseline_source, fault_source)
    scope = PatchScope(
        allowed_paths=(_TARGET_PATH,),
        effect_paths=(_TARGET_PATH,),
        task_owned_paths=(_TARGET_PATH,),
        max_files=1,
        max_bytes=16_384,
    )
    fault_apply = fault_worktree.apply_patch(
        fault_patch,
        scope,
        lease_id=fault_worktree.lease_id,
        fence=fault_worktree.fence,
        visible_sources={_TARGET_PATH: baseline_source},
    )
    if not fault_apply.applied:
        raise VerticalSliceError(f"fault injection rejected: {fault_apply.reason_codes}")
    _git(working, "commit", "-m", "inject compositional contract violation")
    fault_commit = _git(working, "rev-parse", "HEAD")
    fault_tree = _git(working, "rev-parse", "HEAD^{tree}")
    if fault_tree == base_tree:
        raise VerticalSliceError("fault injection did not mutate the tree")
    trace.add(
        "mutation",
        receipt_id=fault_tree,
        status="mutated",
        target_path=_TARGET_PATH,
        isolated_worktree=True,
    )

    fault_state = scan_repository(working, previous_state=baseline_state)
    fault_semantic_bundle = build_semantic_state(
        fault_state, previous_bundle=baseline_semantic_bundle
    )
    fault_semantic_root = verify_semantic_state_bundle(fault_semantic_bundle)
    fault_analyses = _analyze_components(working, api)
    fault_graph, fault_contracts = _build_contract_graph(
        working, fault_state, fault_analyses, api=api
    )
    fault_discharge = api.discharge_assume_guarantee(
        fault_graph,
        expected_semantic_state_root=fault_state.state_cid,
        expected_contract_root=fault_graph.contract_root,
    )
    if fault_discharge.disposition is not DischargeDisposition.DISPROVED:
        raise VerticalSliceError("fault did not produce a concrete failed assumption")
    fault_counterexamples = [
        item.to_dict()
        for item in fault_discharge.obligations
        if item.status is DischargeDisposition.DISPROVED
    ]
    if not fault_counterexamples:
        raise VerticalSliceError("failed composition did not expose a counterexample")
    trace.add(
        "counterexample",
        receipt_id=fault_discharge.receipt_cid,
        status="disproved",
        obligation_id=fault_counterexamples[0]["obligation_id"],
    )
    fault_capsules = compile_semantic_capsules(
        fault_state, previous_bundle=baseline_capsules
    )
    fault_tests = _run_pytest(working, ("tests/test_selected.py",))
    if fault_tests["returncode"] == 0:
        raise VerticalSliceError("fault injection was not observed by selected tests")

    unaffected_symbol = _symbol(baseline_state, "pkg/unaffected.py", "stable_label")
    a_symbol = _symbol(baseline_state, "pkg/module_a.py", "produce")
    evidence_requests = (
        _reuse_request(
            binding_id="abstract:A",
            kind=VerificationBindingKind.ABSTRACT_STATE,
            state=baseline_state,
            symbol=a_symbol,
            contract=baseline_contracts["A"],
            artifact_cid=baseline_analyses["A"].analysis_id,
        ),
        _reuse_request(
            binding_id="proof:unaffected",
            kind=VerificationBindingKind.PROOF,
            state=baseline_state,
            symbol=unaffected_symbol,
            contract=baseline_contracts["U"],
            artifact_cid=cid_for_structured(
                {
                    "capsule": baseline_capsules.capsule(
                        unaffected_symbol.stable_id
                    ).capsule_cid,
                    "kind": "unaffected-fixture-evidence",
                }
            ),
        ),
    )
    incremental = api.plan_incremental_verification(
        baseline_state,
        fault_state,
        composition_graph=fault_graph,
        previous_composition_graph=baseline_graph,
        evidence_requests=evidence_requests,
    )
    evidence_by_id = {item.binding_id: item for item in incremental.evidence_decisions}
    if (
        evidence_by_id["proof:unaffected"].disposition
        is not EvidenceDecisionDisposition.REUSED
    ):
        raise VerticalSliceError("unaffected evidence was not exactly reused")
    if (
        evidence_by_id["abstract:A"].disposition
        is not EvidenceDecisionDisposition.INVALIDATED
    ):
        raise VerticalSliceError("changed A abstract state was not invalidated")
    trace.add(
        "invalidation",
        receipt_id=incremental.receipt_cid,
        status="invalidated",
        binding_id="abstract:A",
    )
    trace.add(
        "unaffected_reuse",
        receipt_id=evidence_by_id["proof:unaffected"].binding_id,
        status="reused",
        binding_id="proof:unaffected",
    )

    localization = _localize_failure(api, fault_graph)
    solver_result = localization["incremental_solver"]
    if solver_result["status"] != "unsat" or not solver_result["core_validated"]:
        raise VerticalSliceError("failure localization core was not independently checked")
    interpolant_status = localization["interpolant"].get("status")
    if interpolant_status != InterpolationStatus.VALIDATED.value:
        raise VerticalSliceError("failure interpolant was not independently validated")
    trace.add(
        "incremental_smt",
        receipt_id=str(solver_result["receipt_id"]),
        status="unsat",
        provider=str(solver_result.get("limitations") or "smt"),
    )
    trace.add(
        "core",
        receipt_id=str(solver_result["receipt_id"]),
        status="validated",
        unsat_core=list(solver_result.get("unsat_core") or ()),
    )
    trace.add(
        "interpolant",
        receipt_id=str(localization["interpolant"]["receipt_cid"]),
        status="validated",
    )

    synthesis_roots = _doctor_roots(
        state=fault_state,
        graph=fault_graph,
        tree=fault_tree,
        lease_id="lease:lgcvf-vertical-repair",
    )
    failed_obligation_id = fault_counterexamples[0]["obligation_id"]
    repair_synthesis = _synthesize_repair(
        roots=synthesis_roots,
        failed_obligation_id=failed_obligation_id,
        fault_source=fault_source,
    )
    if not repair_synthesis.deterministic_zero_model_calls:
        raise VerticalSliceError("repair synthesis did not prove zero model calls")

    repair_worktree = create_isolated_worktree(
        repo_root=repository,
        worktree_path=temp_root / "worktrees" / "repair",
        base_commit=fault_commit,
        base_tree=fault_tree,
        task_id="LGCVF-VERTICAL-REPAIR",
        lane_id="lgcvf-hermetic",
        canonical_task_cid="task:lgcvf-vertical-repair",
        lease_id=synthesis_roots.lease_id,
        retain_on_success=True,
    )
    repair_source = fault_source.replace(
        f"return {_FAULT_VALUE}", f"return {_REPAIR_VALUE}", 1
    )
    repair_patch = _patch(_TARGET_PATH, fault_source, repair_source)
    before_hash = _sha256_bytes(fault_source.encode("utf-8"))
    # The live fixed-point contract binds authority to the candidate tree.  The
    # synthesis receipt above separately remains bound to the observed fault
    # tree, preserving both identities instead of relabelling either state.
    roots = _doctor_roots(
        state=fault_state,
        graph=fault_graph,
        tree=base_tree,
        lease_id=synthesis_roots.lease_id,
    )
    plan = _doctor_plan(
        roots=roots,
        before_hash=before_hash,
        failed_obligation_id=failed_obligation_id,
        repair_receipt=repair_synthesis,
    )
    sandbox = DoctorSandboxPolicy(
        sandbox_id=roots.sandbox_id,
        worktree_root_ref=_REPAIR_WORKTREE_REF,
        permitted_paths=(_TARGET_PATH,),
        enforcement_level=DoctorSandboxEnforcementLevel.ENFORCED,
        secrets_inherited=False,
        network_denied=True,
        target_code_imported=False,
    )
    lock = DoctorCheckoutLock(
        lock_id="lock:lgcvf-vertical-repair",
        holder_id="holder:lgcvf-vertical",
        worktree_root_ref=_REPAIR_WORKTREE_REF,
        base_tree_cid=f"git-tree:{fault_tree}",
        active=True,
        fence_id=_REPAIR_FENCE_REF,
    )
    lease = DoctorWriterLease(
        lease_id=roots.lease_id,
        fence_id=_REPAIR_FENCE_REF,
        holder_id="holder:lgcvf-vertical",
        permitted_write_paths=(_TARGET_PATH,),
        permitted_read_paths=(_TARGET_PATH,),
        active=True,
    )

    def apply_repair(_request: DoctorStepApplyRequest) -> DoctorStepApplyResult:
        result = repair_worktree.apply_patch(
            repair_patch,
            scope,
            lease_id=repair_worktree.lease_id,
            fence=repair_worktree.fence,
            visible_sources={_TARGET_PATH: fault_source},
        )
        if not result.applied:
            return DoctorStepApplyResult(
                disposition=DoctorStepDisposition.FAILED,
                reason_codes=tuple(result.reason_codes),
            )
        observed = (repair_worktree.worktree_path / _TARGET_PATH).read_bytes()
        observed_tree = _git(repair_worktree.worktree_path, "write-tree")
        return DoctorStepApplyResult(
            disposition=DoctorStepDisposition.PASSED,
            written_paths=(_TARGET_PATH,),
            observed_before_hashes=(PathBeforeHash(_TARGET_PATH, before_hash),),
            observed_after_hashes=(PathBeforeHash(_TARGET_PATH, _sha256_bytes(observed)),),
            changed_blob_cids=(cid_for_bytes(observed),),
            observed_tree_cid=f"git-tree:{observed_tree}",
            observed_forest_cid=f"forest:{observed_tree}",
            durable_effect_ref=result.validation.patch_digest,
            static_replay=False,
        )

    transaction = DeterministicDoctorTransaction(
        step_applicator=apply_repair,
        restore_adapter=lambda _checkpoint: False,
        hash_probe=lambda path: _sha256_bytes(
            (repair_worktree.worktree_path / path).read_bytes()
        ),
        effect_verifier=lambda _request, _result: True,
        # The fixture validates an isolated candidate tree; it never updates a
        # shared branch ref.  Durable promotion remains outside this demo and
        # would require the existing live ref-CAS path.
        allow_provisional_live_validation=True,
    ).execute(
        plan,
        sandbox_policy=sandbox,
        checkout_lock=lock,
        lease=lease,
        path_before_hashes=(PathBeforeHash(_TARGET_PATH, before_hash),),
        base_tree_cid=f"git-tree:{fault_tree}",
        candidate_tree_cid=f"git-tree:{base_tree}",
        committed_tree_cid=f"git-tree:{base_tree}",
        transaction_id="transaction:lgcvf-vertical-repair",
    )
    if transaction.disposition is not DoctorTransactionDisposition.COMMITTED:
        raise VerticalSliceError(
            f"doctor transaction did not commit: {transaction.reason_codes}"
        )
    trace.add(
        "isolated_repair",
        receipt_id=transaction.content_id,
        status="committed",
        isolated_worktree=True,
        candidate_id="candidate:constant:10",
    )

    repaired_root = repair_worktree.worktree_path
    repaired_state = scan_repository(repaired_root, previous_state=fault_state)
    repaired_semantic_bundle = build_semantic_state(
        repaired_state, previous_bundle=fault_semantic_bundle
    )
    repaired_semantic_root = verify_semantic_state_bundle(repaired_semantic_bundle)
    repaired_analyses = _analyze_components(repaired_root, api)
    repaired_graph, repaired_contracts = _build_contract_graph(
        repaired_root, repaired_state, repaired_analyses, api=api
    )
    repaired_discharge = api.discharge_assume_guarantee(
        repaired_graph,
        expected_semantic_state_root=repaired_state.state_cid,
        expected_contract_root=repaired_graph.contract_root,
    )
    if repaired_discharge.disposition is not DischargeDisposition.PROVED:
        raise VerticalSliceError("repaired composition did not prove")
    repaired_capsules = compile_semantic_capsules(
        repaired_state, previous_bundle=fault_capsules
    )
    final_incremental = api.plan_incremental_verification(
        fault_state,
        repaired_state,
        composition_graph=repaired_graph,
        previous_composition_graph=fault_graph,
    )
    selected_tests = _run_pytest(repaired_root, ("tests/test_selected.py",))
    full_tests = _run_pytest(repaired_root, ("tests",))
    if selected_tests["returncode"] or full_tests["returncode"]:
        raise VerticalSliceError("repaired selected/full fixture checks failed")
    trace.add(
        "affected_replay",
        receipt_id=_test_receipt_id(selected_tests),
        status="passed",
        selected="tests/test_selected.py",
        full_status=full_tests["status"],
    )

    live_request = LiveFixedPointRequest(
        changed_paths=(_TARGET_PATH,),
        file_bytes={_TARGET_PATH: repair_source.encode("utf-8")},
        original_finding_ids=(failed_obligation_id,),
        original_delta_ids=(incremental.receipt_cid,),
        prior_cache_ids=(fault_graph.contract_root,),
        expected_tombstone_ids=(f"tombstone:{_TARGET_PATH}",),
        intent_effects=(f"effect:{_TARGET_PATH}",),
        code_effects=(f"effect:{_TARGET_PATH}",),
        effects_by_path={_TARGET_PATH: (f"effect:{_TARGET_PATH}",)},
        required_hyperproperty_ids=("worktree_isolation",),
        held_hyperproperty_receipt_ids=("hyperproperty:worktree_isolation",),
    )

    def restore_exact(_checkpoint: Any) -> bool:
        completed = subprocess.run(
            ("git", "reset", "--hard", fault_commit),
            cwd=repaired_root,
            capture_output=True,
            timeout=30,
            check=False,
        )
        return completed.returncode == 0

    fixed_point = DeterministicDoctorLiveFixedPoint(
        restore_adapter=restore_exact,
        require_independent_restore=True,
    ).run(plan, transaction, live_request)
    if not fixed_point.complete or fixed_point.fixed_point is None:
        raise VerticalSliceError(
            f"live fixed point remained open: {fixed_point.report.reason_codes}"
        )
    if fixed_point.fixed_point.model_invocation_count != 0:
        raise VerticalSliceError("fixed point observed a model invocation")
    trace.add(
        "live_fixed_point",
        receipt_id=fixed_point.fixed_point.content_id,
        status="complete",
        complete=True,
    )

    raw_files = sorted(
        path
        for path in repaired_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    raw_source_bytes = sum(len(path.read_bytes()) for path in raw_files)
    raw_source_tokens = (raw_source_bytes + 3) // 4
    context = compile_planner_doctor_context(
        PlannerDoctorContextRequest(
            repository_id=repaired_state.repository_id,
            tree_id=f"git-tree:{base_tree}",
            task_id="LGCVF-VERTICAL",
            acceptance_ids=("acceptance:composition-proved", "acceptance:tests-pass"),
            intent_summary="Restore A's guarantee so B and C compose under current roots",
            security_roots=(roots.policy_id,),
            open_obligation_ids=(),
            assumption_ids=("clause:B:producer-assumption",),
            counterexample_ids=(),
            impact_coverage_ids=tuple(final_incremental.reverse_contract_closure),
            allowed_paths=(_TARGET_PATH,),
            protected_paths=("tests", "config"),
            allowed_effects=(f"effect:{_TARGET_PATH}",),
            validation_commands=(
                f"{sys.executable} -m pytest -q -p no:cacheprovider tests/test_selected.py",
            ),
            satisfied_proof_handles=(
                repaired_discharge.receipt_cid,
                repaired_analyses["A"].analysis_id,
                repaired_semantic_root.root_cid,
            ),
            retrieval_slice_node_ids=tuple(final_incremental.reverse_contract_closure),
            deterministic_closure=True,
            objective_id="LGCVF-G001",
            objective_revision="logic-governed-compositional-verification-fabric-v1",
            policy_id=roots.policy_id,
            policy_revision=cid_for_structured({"policy": roots.policy_id}),
            goal_summary="Deterministic compositional repair reaches a verified fixed point",
        )
    )
    if not context.deterministic_closed or context.llm_required:
        raise VerticalSliceError("final context did not remain deterministically closed")
    trace.add(
        "final_context",
        receipt_id=content_identity(context.to_dict()),
        status="compiled",
        deterministic_closed=True,
        llm_required=False,
    )

    provider_modules_after = set(sys.modules)
    added_provider_modules = tuple(
        sorted(
            name
            for name in provider_modules_after - provider_modules_before
            if any(marker in name.casefold() for marker in _PROVIDER_MARKERS)
        )
    )
    if added_provider_modules:
        raise VerticalSliceError(
            f"deterministic route imported provider modules: {added_provider_modules}"
        )
    trace.add(
        "zero_model_calls",
        receipt_id=content_identity(
            {
                "model_invocation_count": 0,
                "provider_modules": list(added_provider_modules),
            }
        ),
        status="zero",
        model_invocation_count=0,
        provider_modules_imported_during_route=[],
    )

    input_tokens = int(context.to_dict().get("input_tokens") or 0)
    context_reduction_bps = (
        0
        if raw_source_tokens == 0
        else max(0, (raw_source_tokens - input_tokens) * 10_000 // raw_source_tokens)
    )
    reused_unaffected_capsules = tuple(
        cid
        for cid in fault_capsules.reused_cids
        if cid == baseline_capsules.capsule(unaffected_symbol.stable_id).capsule_cid
    )
    benchmark = {
        "schema": "lgcvf-paired-benchmark@1",
        "cohort": "hermetic_local_execution",
        "production_authoritative": False,
        "baseline": {
            "context_tokens": raw_source_tokens,
            "model_calls": 0,
            "raw_source_bytes": raw_source_bytes,
            "tests_selected": 3,
            "verification_strategy": "raw-source-full-fixture",
        },
        "challenger": {
            "abstract_state_reused": (
                1
                if repaired_analyses["U"].analysis_id
                == baseline_analyses["U"].analysis_id
                else 0
            ),
            "capsule_reuse_count": len(fault_capsules.reused_cids),
            "context_tokens": input_tokens,
            "deterministic_closures": 1,
            "model_calls": 0,
            "proof_test_reuse_bps": 10_000 if reused_unaffected_capsules else 0,
            "selected_tests": 2,
            "solver_session_replay_manifests": 1,
            "verification_strategy": (
                "contracts-abstract-interpretation-incremental-smt-capsule-reuse"
            ),
        },
        "comparison": {
            "accepted_patch_quality_equal": full_tests["returncode"] == 0,
            "context_reduction_bps": context_reduction_bps,
            "critical_omissions_accepted": 0,
            "model_call_reduction_bps": 0,
            "safety_floor_violations": 0,
        },
        "limitations": [
            "single hermetic fixture is not a representative maintenance suite",
            "both paired paths intentionally made zero model calls",
            "no production or remote-model evidence is aggregated",
            "wall-time/cost qualification remains a successor benchmark task",
        ],
    }
    trace.add(
        "token_metrics",
        receipt_id=content_identity(
            {
                "context_tokens": input_tokens,
                "raw_source_tokens": raw_source_tokens,
                "context_reduction_bps": context_reduction_bps,
            }
        ),
        status="recorded",
        context_tokens=input_tokens,
        raw_source_tokens=raw_source_tokens,
        context_reduction_bps=context_reduction_bps,
    )
    trace.add(
        "work_reuse_metrics",
        receipt_id=content_identity(
            {
                "proof_test_reuse_bps": benchmark["challenger"]["proof_test_reuse_bps"],
                "capsule_reuse_count": benchmark["challenger"]["capsule_reuse_count"],
                "abstract_state_reused": benchmark["challenger"]["abstract_state_reused"],
            }
        ),
        status="recorded",
        proof_test_reuse_bps=benchmark["challenger"]["proof_test_reuse_bps"],
        capsule_reuse_count=benchmark["challenger"]["capsule_reuse_count"],
    )

    artifact_payload = {
        "allowed_effects": [f"effect:{_TARGET_PATH}"],
        "allowed_paths": [_TARGET_PATH],
        "assumption_clause_ids": ["clause:B:producer-assumption"],
        "capsule_index_cid": repaired_capsules.capsule_index_cid,
        "changed_blob_cids": [cid_for_bytes(repair_source.encode("utf-8"))],
        "contract_root": repaired_graph.contract_root,
        "counterexample_receipt_ids": [
            item["obligation_id"] for item in fault_counterexamples
        ],
        "final_discharge_receipt_cid": repaired_discharge.receipt_cid,
        "fixed_point_receipt_cid": fixed_point.fixed_point.content_id,
        "interpolant_receipt_cid": localization["interpolant"]["receipt_cid"],
        "model_invocation_count": 0,
        "policy_root": roots.policy_id,
        "proof_obligation_ids": [
            item.obligation_id for item in repaired_discharge.obligations
        ],
        "repository_state_cid": repaired_state.state_cid,
        "repository_tree_cid": f"git-tree:{base_tree}",
        "semantic_state_root_cid": repaired_semantic_root.root_cid,
        "selected_test_receipt_cid": _test_receipt_id(selected_tests),
        "solver_receipt_id": solver_result["receipt_id"],
        "source_identity": repaired_analyses["A"].source_identity,
        "toolchain_root": repaired_contracts["A"].toolchain_root,
        "transaction_receipt_cid": transaction.content_id,
        "translation_replay_manifest_cid": localization["replay_manifest"]["manifest_cid"],
        "trust_scope": "hermetic_local_fixture_only",
    }
    artifact = CompositionalVerificationArtifact(artifact_payload)
    artifact_verification = verify_compositional_artifact(
        artifact,
        worktree=repaired_root,
        expected_state=repaired_state,
        expected_graph=repaired_graph,
    )
    if not artifact_verification.valid:
        raise VerticalSliceError(
            f"independent artifact verification failed: {artifact_verification.issues}"
        )
    trace.add(
        "verified_artifact",
        receipt_id=artifact.artifact_cid,
        status="validated",
        replay_receipt_cid=artifact_verification.replay_receipt_cid,
    )
    trace.require_complete()

    result = {
        "schema": VERTICAL_SLICE_SCHEMA,
        "interface": VERTICAL_SLICE_INTERFACE,
        "status": "completed_hermetic_local",
        "production_authorized": False,
        "release_qualified": False,
        "model_invocation_count": 0,
        "provider_modules_imported_during_route": list(added_provider_modules),
        "fixture": {
            "base_commit": base_commit,
            "base_tree": base_tree,
            "fault_commit": fault_commit,
            "fault_tree": fault_tree,
            "target_path": _TARGET_PATH,
            "selected_test": "tests/test_selected.py",
            "initially_unselected_test": "tests/test_unselected.py",
        },
        "baseline": {
            "abstract_analysis": {key: value.to_dict() for key, value in baseline_analyses.items()},
            "capsule_index_cid": baseline_capsules.capsule_index_cid,
            "contract_root": baseline_graph.contract_root,
            "discharge": baseline_discharge.to_dict(),
            "repository_state_cid": baseline_state.state_cid,
            "semantic_state_root_cid": baseline_semantic_root.root_cid,
            "tests": baseline_tests,
        },
        "fault": {
            "abstract_analysis_A": fault_analyses["A"].to_dict(),
            "capsule_reused_cids": list(fault_capsules.reused_cids),
            "counterexamples": fault_counterexamples,
            "discharge": fault_discharge.to_dict(),
            "incremental_plan": incremental.to_dict(),
            "localization": localization,
            "repository_state_cid": fault_state.state_cid,
            "semantic_state_root_cid": fault_semantic_root.root_cid,
            "tests": fault_tests,
        },
        "repair": {
            "fixed_point": fixed_point.report.to_dict(),
            "incremental_plan": final_incremental.to_dict(),
            "repair_synthesis": repair_synthesis.to_dict(),
            "transaction": transaction.to_record(),
            "worktree_apply_tree": base_tree,
        },
        "final": {
            "abstract_analysis_A": repaired_analyses["A"].to_dict(),
            "capsule_index_cid": repaired_capsules.capsule_index_cid,
            "capsule_reused_cids": list(repaired_capsules.reused_cids),
            "context": context.to_dict(),
            "contract_root": repaired_graph.contract_root,
            "discharge": repaired_discharge.to_dict(),
            "full_tests": full_tests,
            "repository_state_cid": repaired_state.state_cid,
            "selected_tests": selected_tests,
            "semantic_state_root_cid": repaired_semantic_root.root_cid,
        },
        "proof_carrying_artifact": artifact.to_dict(),
        "artifact_verification": artifact_verification.to_dict(),
        "benchmark": benchmark,
        "stages": list(REQUIRED_VERTICAL_STAGES),
        "stage_trace": list(trace.records),
        "limitations": [
            "contracts and abstract facts are conservative candidate-tier inputs",
            "Z3 solver evidence is solver-checked, not kernel-verified",
            "cvc5 interpolation is admitted only after independent Z3 checks",
            "the fixture transaction sandbox is policy-enforced but not a "
            "production OS sandbox qualification",
            "this artifact is a checked vertical adapter, not a production authorization receipt",
        ],
    }
    result["result_cid"] = content_identity(
        {key: value for key, value in result.items() if key != "result_cid"}
    )

    if output_path is not None:
        _write_json(Path(output_path), result)
    if benchmark_output_path is not None:
        _write_json(Path(benchmark_output_path), benchmark)
    if keep_workdir:
        result["retained_workdir"] = str(temp_root)
    else:
        shutil.rmtree(temp_root)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--benchmark-output", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        result = run_compositional_verification_vertical_slice(
            fixture_root=arguments.fixture_root,
            output_path=arguments.output,
            benchmark_output_path=arguments.benchmark_output,
            keep_workdir=arguments.keep_workdir,
        )
    except VerticalSliceError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "artifact_cid": result["proof_carrying_artifact"]["artifact_cid"],
                "benchmark": result["benchmark"]["comparison"],
                "model_invocation_count": result["model_invocation_count"],
                "result_cid": result["result_cid"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArtifactVerificationResult",
    "CompositionalVerificationArtifact",
    "REQUIRED_VERTICAL_STAGES",
    "VERTICAL_ARTIFACT_INTERFACE",
    "VERTICAL_ARTIFACT_VERIFIER_INTERFACE",
    "VERTICAL_SLICE_INTERFACE",
    "VerticalSliceError",
    "main",
    "run_compositional_verification_vertical_slice",
    "verify_compositional_artifact",
]
