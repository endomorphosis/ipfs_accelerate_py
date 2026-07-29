"""Tests for the closed, lazy program-analysis capability matrix."""

from __future__ import annotations

import threading
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.ipfs_datasets_program_analysis_provider import (
    CAPABILITY_FAMILY_ORDER,
    DEFAULT_OPTIONAL_ROOT,
    IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_ID,
    PROGRAM_ANALYSIS_CAPABILITY_REPORT_SCHEMA,
    PROGRAM_ANALYSIS_CAPABILITY_SCHEMA,
    CapabilityAuthority,
    CapabilityFamily,
    CapabilityProbeStatus,
    CapabilityReasonCode,
    CapabilitySurface,
    IpfsDatasetsProgramAnalysisProvider,
    ProgramAnalysisCapabilityError,
    ProgramAnalysisCapabilityMatrix,
    ProgramAnalysisCapabilityProbe,
    ProgramAnalysisProbeConfig,
    clear_program_analysis_capability_cache,
    declare_program_analysis_capability_matrix,
    inspect_program_analysis_capability_matrix,
    probe_program_analysis_capabilities,
)


class FakeDiscovery:
    def __init__(
        self,
        *,
        modules: set[str] | None = None,
        executables: dict[str, str] | None = None,
        import_map: dict[str, object] | None = None,
        slow_modules: set[str] | None = None,
        sleep: float = 0.0,
    ) -> None:
        self.modules = modules or set()
        self.executables = executables or {}
        self.import_map = import_map or {}
        self.slow_modules = slow_modules or set()
        self.sleep = sleep
        self.package_calls: list[str] = []
        self.import_calls: list[str] = []
        self.executable_calls: list[str] = []

    def find_spec(self, module: str) -> object | None:
        self.package_calls.append(module)
        if module in self.modules or module in self.import_map:
            return SimpleNamespace(origin=f"/python/{module.replace('.', '/')}.py")
        return None

    def which(self, executable: str) -> str | None:
        self.executable_calls.append(executable)
        return self.executables.get(executable)

    def importer(self, module: str) -> object:
        self.import_calls.append(module)
        if module in self.slow_modules:
            import time

            time.sleep(self.sleep)
        if module not in self.import_map:
            raise ModuleNotFoundError(module)
        return self.import_map[module]


class MutableClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _cid_module() -> types.ModuleType:
    mod = types.ModuleType("ipfs_datasets_py.utils.cid_utils")

    def canonical_dag_json_bytes(obj):
        import json

        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def cid_for_bytes(data, *, base="base32", codec="raw", mh_type="sha2-256", version=1):
        import hashlib

        digest = hashlib.sha256(data).hexdigest()
        return f"bafy{digest[:50]}"

    def cid_for_dag_json(obj, *, base="base32", mh_type="sha2-256", version=1):
        return cid_for_bytes(
            canonical_dag_json_bytes(obj), base=base, codec="dag-json", mh_type=mh_type, version=version
        )

    def cid_for_obj(obj, *, base="base32", codec="raw", mh_type="sha2-256", version=1):
        return cid_for_bytes(
            canonical_dag_json_bytes(obj), base=base, codec=codec, mh_type=mh_type, version=version
        )

    def validate_cid(
        value,
        *,
        codecs=("raw", "dag-json"),
        mh_type="sha2-256",
        version=1,
        base="base32",
    ):
        if not isinstance(value, str) or not value or value != value.lower():
            raise ValueError("CID must be a nonempty lowercase string")
        if value.startswith("Qm") or ":" in value or value.startswith("tree:"):
            raise ValueError("pseudo or legacy CID rejected")
        if not value.startswith("bafy") and not value.startswith("baga"):
            raise ValueError("CID is not decodable")
        return value

    mod.canonical_dag_json_bytes = canonical_dag_json_bytes
    mod.cid_for_bytes = cid_for_bytes
    mod.cid_for_dag_json = cid_for_dag_json
    mod.cid_for_obj = cid_for_obj
    mod.validate_cid = validate_cid
    return mod


def _ir_claims_module() -> types.ModuleType:
    mod = types.ModuleType("ipfs_datasets_py.logic.ir_core.claims")
    mod.IRClaim = object
    mod.IRAssumption = object
    mod.IRObligation = object
    mod.ClaimValidationError = ValueError
    mod.IR_CLAIM_SCHEMA_VERSION = "ir-claim/v1"
    return mod


def _ir_protocols_module() -> types.ModuleType:
    from enum import Enum

    mod = types.ModuleType("ipfs_datasets_py.logic.ir_core.protocols")

    class AuthorityKind(str, Enum):
        THEOREM_PROOF = "theorem_proof"
        SATISFIABILITY = "satisfiability"

    class QueryKind(str, Enum):
        THEOREM_PROOF = "theorem_proof"

    class AttemptStatus(str, Enum):
        SUCCEEDED = "succeeded"

    mod.AuthorityKind = AuthorityKind
    mod.QueryKind = QueryKind
    mod.AttemptStatus = AttemptStatus
    mod.ProtocolValidationError = ValueError
    mod.AuthorityMismatchError = ValueError
    mod.BACKEND_CAPABILITIES_SCHEMA_VERSION = "proof-backend-capabilities/v1"
    return mod


def _graphrag_module() -> types.ModuleType:
    mod = types.ModuleType(
        "ipfs_datasets_py.search.graphrag_integration.graphrag_integration"
    )

    class GraphRAGQueryEngine:
        def query(self, query_text: str, top_k: int = 10, max_nodes_visited=None):
            return {"results": []}

    mod.GraphRAGQueryEngine = GraphRAGQueryEngine
    return mod


def _ast_module() -> types.ModuleType:
    mod = types.ModuleType(
        "ipfs_datasets_py.logic.security_models.crypto_exchange.extractors."
        "python_ast_extractor"
    )

    class PythonASTExtractor:
        def extract_from_source(self, source: str, *, module_path: str = "<memory>"):
            return {"functions": [], "classes": []}

    mod.PythonASTExtractor = PythonASTExtractor
    return mod


def _zkp_backends_module() -> types.ModuleType:
    mod = types.ModuleType("ipfs_datasets_py.logic.zkp.backends")
    mod._BACKEND_METADATA = {
        "simulated": {"description": "educational"},
        "groth16": {"description": "real"},
        "provekit": {"description": "real"},
    }

    def get_backend(backend: str = "simulated"):
        return SimpleNamespace(backend_id=backend or "simulated")

    mod.get_backend = get_backend
    return mod


def _zkp_circuits_module() -> types.ModuleType:
    return types.ModuleType("ipfs_datasets_py.logic.zkp.circuits")


def _full_import_map() -> dict[str, object]:
    root = DEFAULT_OPTIONAL_ROOT
    return {
        f"{root}.utils.cid_utils": _cid_module(),
        "multiformats": types.ModuleType("multiformats"),
        f"{root}.search.graphrag_integration.graphrag_integration": _graphrag_module(),
        f"{root}.search.graph_query": types.ModuleType(f"{root}.search.graph_query"),
        f"{root}.knowledge_graphs": types.ModuleType(f"{root}.knowledge_graphs"),
        f"{root}.logic.ir_core.claims": _ir_claims_module(),
        f"{root}.logic.ir_core.protocols": _ir_protocols_module(),
        f"{root}.logic.security_models.crypto_exchange.extractors.python_ast_extractor": _ast_module(),
        f"{root}.logic.zkp.backends": _zkp_backends_module(),
        f"{root}.logic.zkp.circuits": _zkp_circuits_module(),
        f"{root}.logic.external_provers.smt.cvc5_prover_bridge": types.ModuleType(
            "cvc5_bridge"
        ),
        f"{root}.logic.external_provers.smt.z3_prover_bridge": types.ModuleType(
            "z3_bridge"
        ),
        "cvc5": SimpleNamespace(Solver=object, __version__="1.0-test"),
        "z3": SimpleNamespace(Solver=object, __version__="4.0-test"),
    }


def _full_modules() -> set[str]:
    return set(_full_import_map())


def test_cold_import_construction_and_matrix_never_import_optional_code() -> None:
    calls: list[str] = []

    def importer(name: str):
        calls.append(name)
        raise AssertionError("cold path imported optional code")

    def find_spec(name: str):
        calls.append(f"spec:{name}")
        raise AssertionError("cold path probed package specs")

    provider = IpfsDatasetsProgramAnalysisProvider(
        importer=importer,
        find_spec=find_spec,
        which=lambda name: (_ for _ in ()).throw(
            AssertionError("cold path looked up executables")
        ),
    )
    state_before = {name: id(value) for name, value in vars(provider).items()}

    with ThreadPoolExecutor(max_workers=8) as executor:
        matrices = tuple(executor.map(lambda _: provider.capabilities(), range(32)))
    first = matrices[0]
    second = provider.capability()
    pure = declare_program_analysis_capability_matrix()
    inspected = inspect_program_analysis_capability_matrix(pure.to_dict())

    assert calls == []
    assert all(item == first for item in matrices)
    assert first == second
    assert first.to_dict() == pure.to_dict()
    assert inspected.matrix_id == pure.matrix_id
    assert first.lazy_import is True
    assert first.probed is False
    assert first.imported is False
    assert first.non_authoritative is True
    assert first.completion_authority is False
    assert first.proof_attempted is False
    assert first.proof_success is False
    assert tuple(item.value for item in first.families) == CAPABILITY_FAMILY_ORDER
    assert first.schema_version == PROGRAM_ANALYSIS_CAPABILITY_SCHEMA
    assert first.provider_id == IPFS_DATASETS_PROGRAM_ANALYSIS_PROVIDER_ID
    assert {name: id(value) for name, value in vars(provider).items()} == state_before
    assert ProgramAnalysisCapabilityMatrix.from_dict(first.to_dict()) == first


def test_unavailable_optional_modules_degrade_explicitly() -> None:
    discovery = FakeDiscovery(modules=set(), import_map={})
    probe = ProgramAnalysisCapabilityProbe(
        ProgramAnalysisProbeConfig(cache_ttl_seconds=0),
        find_spec=discovery.find_spec,
        which=discovery.which,
        importer=discovery.importer,
    )
    report = probe.probe()

    assert report.schema_version == PROGRAM_ANALYSIS_CAPABILITY_REPORT_SCHEMA
    assert report.non_authoritative is True
    assert report.completion_authority is False
    assert report.proof_attempted is False
    assert report.proof_success is False
    assert report.bounded is True
    assert report.overall_status is CapabilityProbeStatus.UNAVAILABLE
    assert tuple(item.family.value for item in report.families) == CAPABILITY_FAMILY_ORDER
    for family in report.families:
        assert family.status in {
            CapabilityProbeStatus.UNAVAILABLE,
            CapabilityProbeStatus.TIMED_OUT,
        }
        assert family.completion_authority is False
    # Missing packages fail closed at discovery; imports are not attempted.
    assert discovery.package_calls
    assert discovery.import_calls == []
    assert any(
        surface.reason_code is CapabilityReasonCode.PACKAGE_MISSING
        for family in report.families
        for surface in family.surfaces
    )
    assert report.diagnostics["package_presence_is_not_capability"] is True
    assert report.diagnostics["simulated_zkp_authority"] is False


def test_incompatible_method_signatures_are_rejected() -> None:
    root = DEFAULT_OPTIONAL_ROOT
    broken_cid = types.ModuleType(f"{root}.utils.cid_utils")
    # Wrong signatures / missing required callables.
    broken_cid.canonical_dag_json_bytes = lambda: b"{}"
    broken_cid.cid_for_bytes = lambda: "x"
    broken_cid.cid_for_dag_json = lambda: "x"
    broken_cid.cid_for_obj = lambda: "x"
    broken_cid.validate_cid = lambda: "x"

    broken_ast = types.ModuleType(
        f"{root}.logic.security_models.crypto_exchange.extractors.python_ast_extractor"
    )

    class BrokenExtractor:
        def extract_from_source(self):  # missing required `source`
            return {}

    broken_ast.PythonASTExtractor = BrokenExtractor

    import_map = {
        f"{root}.utils.cid_utils": broken_cid,
        f"{root}.logic.security_models.crypto_exchange.extractors.python_ast_extractor": broken_ast,
        f"{root}.logic.ir_core.claims": types.ModuleType("claims"),  # missing exports
        f"{root}.logic.ir_core.protocols": types.ModuleType("protocols"),
        f"{root}.logic.zkp.backends": types.ModuleType("backends"),  # missing get_backend
        f"{root}.logic.zkp.circuits": _zkp_circuits_module(),
    }
    discovery = FakeDiscovery(modules=set(import_map), import_map=import_map)
    probe = ProgramAnalysisCapabilityProbe(
        ProgramAnalysisProbeConfig(cache_ttl_seconds=0, run_strict_cid_canary=False),
        find_spec=discovery.find_spec,
        which=discovery.which,
        importer=discovery.importer,
    )
    report = probe.probe()

    cid = report.family(CapabilityFamily.STRICT_CID)
    assert cid.status is CapabilityProbeStatus.INCOMPATIBLE
    assert any(
        surface.reason_code is CapabilityReasonCode.SIGNATURE_INCOMPATIBLE
        for surface in cid.surfaces
    )

    ast = report.family(CapabilityFamily.AST_PRODUCERS)
    assert ast.status is CapabilityProbeStatus.INCOMPATIBLE

    ir = report.family(CapabilityFamily.IR_CORE)
    assert ir.status is CapabilityProbeStatus.INCOMPATIBLE

    zkp = report.family(CapabilityFamily.ZKP)
    assert zkp.status in {
        CapabilityProbeStatus.INCOMPATIBLE,
        CapabilityProbeStatus.UNAVAILABLE,
        CapabilityProbeStatus.SIMULATED,
    }


def test_partial_matrix_when_only_some_families_are_ready() -> None:
    root = DEFAULT_OPTIONAL_ROOT
    import_map = {
        f"{root}.utils.cid_utils": _cid_module(),
        "multiformats": types.ModuleType("multiformats"),
        f"{root}.logic.ir_core.claims": _ir_claims_module(),
        # protocols intentionally missing -> partial IR
        f"{root}.logic.security_models.crypto_exchange.extractors.python_ast_extractor": _ast_module(),
        f"{root}.logic.zkp.backends": _zkp_backends_module(),
        f"{root}.logic.zkp.circuits": _zkp_circuits_module(),
        # solvers: only cvc5 executable, no z3
    }
    discovery = FakeDiscovery(
        modules=set(import_map),
        import_map=import_map,
        executables={"cvc5": "/usr/bin/cvc5"},
    )
    probe = ProgramAnalysisCapabilityProbe(
        ProgramAnalysisProbeConfig(cache_ttl_seconds=0),
        find_spec=discovery.find_spec,
        which=discovery.which,
        importer=discovery.importer,
    )
    report = probe.probe()

    assert report.overall_status is CapabilityProbeStatus.PARTIAL
    assert report.family(CapabilityFamily.STRICT_CID).status is CapabilityProbeStatus.AVAILABLE
    ir = report.family(CapabilityFamily.IR_CORE)
    assert ir.status is CapabilityProbeStatus.PARTIAL
    solvers = report.family(CapabilityFamily.SOLVERS)
    assert solvers.status in {
        CapabilityProbeStatus.PARTIAL,
        CapabilityProbeStatus.AVAILABLE,
        CapabilityProbeStatus.DEGRADED,
    }
    ast = report.family(CapabilityFamily.AST_PRODUCERS)
    assert ast.status is CapabilityProbeStatus.PARTIAL
    assert any(surface.metadata.get("limited") for surface in ast.surfaces)
    zkp = report.family(CapabilityFamily.ZKP)
    assert zkp.status is CapabilityProbeStatus.SIMULATED
    assert zkp.authority is CapabilityAuthority.ZKP_DIAGNOSTIC
    assert any(
        surface.reason_code
        is CapabilityReasonCode.SIMULATED_ZKP_AUTHORITY_REJECTED
        for surface in zkp.surfaces
    )


def test_timeout_marks_probe_limited_and_does_not_hang() -> None:
    clock = MutableClock(100.0)
    root = DEFAULT_OPTIONAL_ROOT
    import_map = _full_import_map()
    discovery = FakeDiscovery(
        modules=set(import_map),
        import_map=import_map,
        slow_modules={f"{root}.utils.cid_utils"},
        sleep=0.2,
    )

    def monotonic() -> float:
        # Force remaining budget to be tiny once the slow import starts.
        return clock.value

    probe = ProgramAnalysisCapabilityProbe(
        ProgramAnalysisProbeConfig(timeout_seconds=0.05, cache_ttl_seconds=0, max_checks=10_000),
        find_spec=discovery.find_spec,
        which=discovery.which,
        importer=discovery.importer,
        monotonic=monotonic,
        wall_clock=lambda: 1_700_000_000.0,
    )
    report = probe.probe()
    assert report.bounded is True
    # Either overall timed out or individual surfaces recorded timeout/limit.
    timed = report.overall_status is CapabilityProbeStatus.TIMED_OUT or any(
        surface.status is CapabilityProbeStatus.TIMED_OUT
        or surface.reason_code
        in {
            CapabilityReasonCode.PROBE_TIMEOUT,
            CapabilityReasonCode.PROBE_LIMIT,
        }
        for family in report.families
        for surface in family.surfaces
    )
    assert timed
    assert report.diagnostics.get("probe_limited") in {True, False}


def test_probe_check_limit_is_enforced() -> None:
    discovery = FakeDiscovery(
        modules=_full_modules(),
        import_map=_full_import_map(),
        executables={"cvc5": "/bin/cvc5", "z3": "/bin/z3"},
    )
    probe = ProgramAnalysisCapabilityProbe(
        ProgramAnalysisProbeConfig(
            timeout_seconds=30.0,
            cache_ttl_seconds=0,
            max_checks=3,
        ),
        find_spec=discovery.find_spec,
        which=discovery.which,
        importer=discovery.importer,
    )
    report = probe.probe()
    assert any(
        surface.reason_code is CapabilityReasonCode.PROBE_LIMIT
        or surface.metadata.get("probe_limited")
        for family in report.families
        for surface in family.surfaces
    )


def test_package_presence_alone_is_not_capability() -> None:
    root = DEFAULT_OPTIONAL_ROOT
    # Discoverable but import raises, or import succeeds without required API.
    empty_cid = types.ModuleType(f"{root}.utils.cid_utils")
    discovery = FakeDiscovery(
        modules={f"{root}.utils.cid_utils", "multiformats"},
        import_map={f"{root}.utils.cid_utils": empty_cid},
    )
    probe = ProgramAnalysisCapabilityProbe(
        ProgramAnalysisProbeConfig(cache_ttl_seconds=0, run_strict_cid_canary=False),
        find_spec=discovery.find_spec,
        which=discovery.which,
        importer=discovery.importer,
    )
    report = probe.probe()
    cid = report.family(CapabilityFamily.STRICT_CID)
    assert cid.status is not CapabilityProbeStatus.AVAILABLE
    assert any(
        surface.reason_code
        in {
            CapabilityReasonCode.CALLABLE_MISSING,
            CapabilityReasonCode.SIGNATURE_INCOMPATIBLE,
            CapabilityReasonCode.PACKAGE_PRESENCE_ONLY,
        }
        for surface in cid.surfaces
    )


def test_pseudo_cids_are_rejected_by_strict_canary() -> None:
    root = DEFAULT_OPTIONAL_ROOT
    leaky = _cid_module()

    def validate_cid(value, **kwargs):
        # Incorrectly accepts CIDv0 / pseudo forms.
        return value

    leaky.validate_cid = validate_cid
    import_map = {f"{root}.utils.cid_utils": leaky, "multiformats": types.ModuleType("multiformats")}
    discovery = FakeDiscovery(modules=set(import_map), import_map=import_map)
    probe = ProgramAnalysisCapabilityProbe(
        ProgramAnalysisProbeConfig(cache_ttl_seconds=0),
        find_spec=discovery.find_spec,
        which=discovery.which,
        importer=discovery.importer,
    )
    report = probe.probe()
    cid = report.family(CapabilityFamily.STRICT_CID)
    assert cid.status is CapabilityProbeStatus.REJECTED
    assert any(
        surface.reason_code is CapabilityReasonCode.PSEUDO_CID_ACCEPTED
        for surface in cid.surfaces
    )


def test_simulated_zkp_authority_cannot_be_enabled() -> None:
    with pytest.raises(ProgramAnalysisCapabilityError):
        ProgramAnalysisProbeConfig(allow_simulated_zkp_authority=True)


def test_unbounded_graphrag_query_is_rejected() -> None:
    root = DEFAULT_OPTIONAL_ROOT
    mod = types.ModuleType(
        f"{root}.search.graphrag_integration.graphrag_integration"
    )

    class GraphRAGQueryEngine:
        def query(self, query_text: str):
            return {"results": ["x"] * 10_000}

    mod.GraphRAGQueryEngine = GraphRAGQueryEngine
    import_map = {
        f"{root}.search.graphrag_integration.graphrag_integration": mod,
    }
    discovery = FakeDiscovery(modules=set(import_map), import_map=import_map)
    probe = ProgramAnalysisCapabilityProbe(
        ProgramAnalysisProbeConfig(cache_ttl_seconds=0),
        find_spec=discovery.find_spec,
        which=discovery.which,
        importer=discovery.importer,
    )
    report = probe.probe()
    graph = report.family(CapabilityFamily.GRAPHRAG)
    assert any(
        surface.reason_code is CapabilityReasonCode.UNBOUNDED_OUTPUT
        for surface in graph.surfaces
    )
    assert graph.status in {
        CapabilityProbeStatus.REJECTED,
        CapabilityProbeStatus.UNAVAILABLE,
        CapabilityProbeStatus.DEGRADED,
    }


def test_current_probe_reflects_live_environment_as_diagnostics_not_constants() -> None:
    clear_program_analysis_capability_cache()
    report = probe_program_analysis_capabilities(force_refresh=True)

    assert report.schema_version == PROGRAM_ANALYSIS_CAPABILITY_REPORT_SCHEMA
    assert report.diagnostics["observations_are_diagnostics"] is True
    assert report.diagnostics["package_presence_is_not_capability"] is True
    assert report.diagnostics["simulated_zkp_authority"] is False
    assert report.non_authoritative is True
    assert report.completion_authority is False
    assert report.matrix.probed is True
    assert tuple(item.family.value for item in report.families) == CAPABILITY_FAMILY_ORDER

    solvers = report.family(CapabilityFamily.SOLVERS)
    # Live observation: cvc5 CLI is present in this environment; z3 may not be.
    # The matrix must encode that as diagnostics, never hard-coded constants.
    cvc5_exe = next(
        (
            surface
            for surface in solvers.surfaces
            if surface.surface_id == "solver.executable.cvc5"
        ),
        None,
    )
    z3_exe = next(
        (
            surface
            for surface in solvers.surfaces
            if surface.surface_id == "solver.executable.z3"
        ),
        None,
    )
    assert cvc5_exe is not None and z3_exe is not None
    # Do not assert fixed availability; assert the report is structured and
    # each surface carries an explicit reason rather than a silent default.
    assert cvc5_exe.reason
    assert z3_exe.reason
    assert cvc5_exe.status in set(CapabilityProbeStatus)
    assert z3_exe.status in set(CapabilityProbeStatus)

    zkp = report.family(CapabilityFamily.ZKP)
    assert zkp.authority in {
        CapabilityAuthority.ZKP_DIAGNOSTIC,
        CapabilityAuthority.NONE,
    }
    if any(
        surface.status is CapabilityProbeStatus.SIMULATED for surface in zkp.surfaces
    ):
        assert any(
            surface.reason_code
            is CapabilityReasonCode.SIMULATED_ZKP_AUTHORITY_REJECTED
            for surface in zkp.surfaces
        )

    # Round-trip stability.
    restored = type(report).from_dict(report.to_dict())
    assert restored.report_id == report.report_id
    assert restored.overall_status is report.overall_status


def test_capability_surface_never_claims_proof_or_completion() -> None:
    surface = CapabilitySurface(
        surface_id="example",
        status=CapabilityProbeStatus.AVAILABLE,
        reason_code=CapabilityReasonCode.AVAILABLE,
        reason="example surface",
        authority=CapabilityAuthority.DIAGNOSTIC,
    )
    payload = surface.to_dict()
    assert payload["proof_attempted"] is False
    assert payload["proof_success"] is False
    assert payload["completion_authority"] is False
    assert CapabilitySurface.from_dict(payload).to_dict() == payload


def test_closed_matrix_rejects_unknown_or_reordered_families() -> None:
    with pytest.raises(ProgramAnalysisCapabilityError):
        ProgramAnalysisCapabilityMatrix(
            families=(CapabilityFamily.ZKP, CapabilityFamily.STRICT_CID)
        )
    with pytest.raises(ProgramAnalysisCapabilityError):
        ProgramAnalysisProbeConfig(families=("not_a_family",))  # type: ignore[arg-type]


def test_provider_probe_is_thread_safe_for_matrix_declaration() -> None:
    provider = IpfsDatasetsProgramAnalysisProvider(
        ProgramAnalysisProbeConfig(cache_ttl_seconds=0),
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
        find_spec=lambda name: None,
        which=lambda name: None,
    )
    barrier = threading.Barrier(8)
    results: list[ProgramAnalysisCapabilityMatrix] = []

    def worker() -> None:
        barrier.wait(timeout=5)
        results.append(provider.capabilities())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert len(results) == 8
    assert all(item.matrix_id == results[0].matrix_id for item in results)
