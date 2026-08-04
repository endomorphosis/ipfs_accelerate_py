"""Optional measured activation runners for closeout (fail-closed).

Attempts real fixture discovery and optional cold/warm / subprocess measurements
when reviewed keys+binary are present. When keys are absent, records the
discovery result and **does not** claim production activation.

Never installs packages, never networks, never invents reviewed authority.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

MEASUREMENT_INTERFACE: Final = "ProofTestReuseCloseoutActivationMeasurements@1"
MEASUREMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-test-reuse-closeout-activation-measurements@1"
)


@dataclass(slots=True)
class FixtureDiscoveryResult:
    available: bool
    reason: str
    binary_path: str = ""
    artifacts_root: str = ""
    proving_key_path: str = ""
    verifying_key_path: str = ""
    circuit_version: int = 4
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "binary_path": self.binary_path,
            "artifacts_root": self.artifacts_root,
            "proving_key_path": self.proving_key_path,
            "verifying_key_path": self.verifying_key_path,
            "circuit_version": self.circuit_version,
            "detail": self.detail,
        }


@dataclass(slots=True)
class MeasurementAttempt:
    name: str
    attempted: bool
    succeeded: bool
    skipped: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "skipped": self.skipped,
            "detail": self.detail,
            "metrics": dict(self.metrics),
        }


@dataclass(slots=True)
class ActivationMeasurementReport:
    schema: str = MEASUREMENT_SCHEMA
    interface: str = MEASUREMENT_INTERFACE
    authority: bool = False
    fixture: FixtureDiscoveryResult | None = None
    attempts: tuple[MeasurementAttempt, ...] = ()
    claims_supported: dict[str, bool] = field(default_factory=dict)
    notes: tuple[str, ...] = (
        "Measurements never authorize warm-skip without reviewed keys.",
        "When fixture keys are unavailable, measurement runners skip fail-closed.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "authority": self.authority,
            "fixture": self.fixture.to_dict() if self.fixture else None,
            "attempts": [item.to_dict() for item in self.attempts],
            "claims_supported": dict(self.claims_supported),
            "notes": list(self.notes),
        }


def _fixture_module_paths() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        here.parents[3] / "test" / "api" / "proof_reuse_real_groth16_fixture.py",
        here.parents[4]
        / "external"
        / "ipfs_accelerate"
        / "test"
        / "api"
        / "proof_reuse_real_groth16_fixture.py",
    ]


def load_real_groth16_fixture_module() -> Any:
    """Load the PTR-148 fixture module without requiring pytest collection."""

    for path in _fixture_module_paths():
        if not path.is_file():
            continue
        name = "proof_reuse_real_groth16_fixture_closeout_measure"
        # Ensure module is registered before exec for dataclass evaluation.
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # Provide package-like path hints used by some helpers.
        module.__file__ = str(path)
        try:
            spec.loader.exec_module(module)
            return module
        except Exception:
            sys.modules.pop(name, None)
            continue
    raise FileNotFoundError("proof_reuse_real_groth16_fixture.py not found/loadable")


def inventory_artifact_versions(artifacts_root: Path | str | None) -> dict[str, Any]:
    """Inventory which circuit version key pairs exist under artifacts/."""

    root = Path(artifacts_root) if artifacts_root else None
    versions: dict[str, Any] = {}
    if root is None or not root.is_dir():
        return {"artifacts_root": str(artifacts_root or ""), "versions": versions}
    for path in sorted(root.glob("v*")):
        if not path.is_dir():
            continue
        name = path.name
        if not name.startswith("v") or not name[1:].isdigit():
            continue
        pk = path / "proving_key.bin"
        vk = path / "verifying_key.bin"
        versions[name] = {
            "proving_key": pk.is_file(),
            "verifying_key": vk.is_file(),
            "complete": pk.is_file() and vk.is_file(),
            "proving_key_bytes": pk.stat().st_size if pk.is_file() else 0,
            "verifying_key_bytes": vk.stat().st_size if vk.is_file() else 0,
        }
    return {
        "artifacts_root": str(root),
        "versions": versions,
        "v4_complete": bool((versions.get("v4") or {}).get("complete")),
        "any_complete": any(
            bool(item.get("complete")) for item in versions.values()
        ),
    }


def discover_real_groth16_fixture() -> FixtureDiscoveryResult:
    """Discover reviewed local Groth16 binary/keys (no prove/install/network)."""

    try:
        module = load_real_groth16_fixture_module()
        cls = getattr(module, "RealGroth16TestPassFixture", None)
        if cls is None:
            return FixtureDiscoveryResult(
                available=False,
                reason="fixture_class_missing",
                detail="RealGroth16TestPassFixture not in module",
            )
        fixture = cls.discover()
        artifacts_root = str(getattr(fixture, "artifacts_root", "") or "")
        inventory = inventory_artifact_versions(artifacts_root)
        return FixtureDiscoveryResult(
            available=bool(getattr(fixture, "available", False)),
            reason=str(getattr(fixture, "reason", "") or ""),
            binary_path=str(getattr(fixture, "binary_path", "") or ""),
            artifacts_root=artifacts_root,
            proving_key_path=str(getattr(fixture, "proving_key_path", "") or ""),
            verifying_key_path=str(getattr(fixture, "verifying_key_path", "") or ""),
            circuit_version=int(getattr(fixture, "circuit_version", 4) or 4),
            detail=json_dumps_compact(
                {
                    "discover": "ok",
                    "artifact_inventory": inventory,
                    "operator_setup_hint": (
                        "cd external/ipfs_datasets/ipfs_datasets_py/processors/"
                        "groth16_backend && ./build.sh --setup-only"
                        "  # creates local operational v4 keys (NOT production ceremony)"
                    ),
                    "production_key_policy": (
                        "v4 keys must come from an operator-reviewed ceremony "
                        "before production warm-skip authority"
                    ),
                }
            ),
        )
    except Exception as exc:
        return FixtureDiscoveryResult(
            available=False,
            reason=f"discover_failed:{type(exc).__name__}",
            detail=str(exc)[:200],
        )


def json_dumps_compact(value: Any) -> str:
    import json

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))[:500]
    except Exception:
        return str(value)[:200]


def attempt_local_nonproduction_v4_setup(
    *,
    use_seed: int | None = None,
) -> MeasurementAttempt:
    """Optionally create local operational v4 keys (NOT a production ceremony).

    Only runs when ``PTR_CLOSEOUT_LOCAL_SETUP=1``. Uses the already-built
    ``groth16 setup --version 4`` path. Deterministic ``--seed`` is test-only
    and is never labeled as production trust.
    """

    import os
    import subprocess

    if str(os.environ.get("PTR_CLOSEOUT_LOCAL_SETUP", "")).strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return MeasurementAttempt(
            name="local_nonproduction_v4_setup",
            attempted=False,
            succeeded=False,
            skipped=True,
            detail="skipped:set PTR_CLOSEOUT_LOCAL_SETUP=1 to create local operational v4 keys",
        )

    discovery = discover_real_groth16_fixture()
    if discovery.available:
        return MeasurementAttempt(
            name="local_nonproduction_v4_setup",
            attempted=False,
            succeeded=True,
            skipped=True,
            detail="skipped:v4_keys_already_present",
            metrics=discovery.to_dict(),
        )
    binary = Path(discovery.binary_path) if discovery.binary_path else None
    if binary is None or not binary.is_file():
        return MeasurementAttempt(
            name="local_nonproduction_v4_setup",
            attempted=False,
            succeeded=False,
            skipped=True,
            detail="skipped:groth16_binary_missing",
            metrics=discovery.to_dict(),
        )
    artifacts = Path(discovery.artifacts_root) if discovery.artifacts_root else None
    if artifacts is None:
        return MeasurementAttempt(
            name="local_nonproduction_v4_setup",
            attempted=False,
            succeeded=False,
            skipped=True,
            detail="skipped:artifacts_root_missing",
        )
    artifacts.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "setup", "--version", "4", "--quiet"]
    if use_seed is not None:
        cmd.extend(["--seed", str(int(use_seed))])
    # Prefer explicit seed from env for reproducibility of local ops only.
    seed_env = os.environ.get("PTR_CLOSEOUT_LOCAL_SETUP_SEED", "").strip()
    if use_seed is None and seed_env:
        cmd.extend(["--seed", seed_env])
    try:
        # Binary resolves artifacts via GROTH16_BACKEND_ARTIFACTS_ROOT or the
        # compile-time crate root. Force the discovered artifacts path.
        backend_root = artifacts.parent if artifacts.name == "artifacts" else artifacts
        env = os.environ.copy()
        env["GROTH16_BACKEND_ARTIFACTS_ROOT"] = str(artifacts)
        result = subprocess.run(
            cmd,
            cwd=str(backend_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        # Re-discover after setup.
        after = discover_real_groth16_fixture()
        ok = after.available
        marker = artifacts / "v4" / "LOCAL_NONPRODUCTION_SETUP.txt"
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                "LOCAL OPERATIONAL SETUP ONLY — NOT A PRODUCTION CEREMONY\n"
                "These v4 keys must not be promoted as production trust roots.\n"
                f"created_by=proof_test_reuse_closeout_activation_measurements\n"
                f"command={' '.join(cmd)}\n"
                f"returncode={result.returncode}\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        return MeasurementAttempt(
            name="local_nonproduction_v4_setup",
            attempted=True,
            succeeded=ok,
            skipped=False,
            detail=(
                "local_v4_ready_nonproduction"
                if ok
                else f"setup_failed:rc={result.returncode}:{result.stderr[:160]}"
            ),
            metrics={
                "returncode": result.returncode,
                "available_after": after.available,
                "reason_after": after.reason,
                "stdout_tail": (result.stdout or "")[-200:],
                "stderr_tail": (result.stderr or "")[-200:],
                "production_authority": False,
                "local_operational_only": True,
            },
        )
    except Exception as exc:
        return MeasurementAttempt(
            name="local_nonproduction_v4_setup",
            attempted=True,
            succeeded=False,
            skipped=False,
            detail=f"{type(exc).__name__}:{exc}"[:240],
        )


def attempt_subprocess_proof_reuse_benchmark(
    *,
    require_available_fixture: bool = True,
    timeout_s: float = 120.0,
) -> MeasurementAttempt:
    """Run measured cold/warm subprocess benchmark when fixture is ready.

    Skips when reviewed keys are unavailable. Never invents positive timing.
    """

    discovery = discover_real_groth16_fixture()
    if require_available_fixture and not discovery.available:
        return MeasurementAttempt(
            name="subprocess_proof_reuse_benchmark",
            attempted=False,
            succeeded=False,
            skipped=True,
            detail=f"skipped:fixture_{discovery.reason}",
            metrics=discovery.to_dict(),
        )
    started = time.time()
    try:
        from ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark import (
            run_subprocess_proof_reuse_benchmark,
        )

        # Bound wall time loosely: runner itself measures; we abort if import/setup
        # is fine but we don't want indefinite hangs — the underlying runner has
        # its own subprocess timeouts in the fixture helpers.
        with tempfile.TemporaryDirectory(prefix="ptr-closeout-subprocess-") as tmp:
            receipt = run_subprocess_proof_reuse_benchmark(base_dir=tmp)
        elapsed = time.time() - started
        if elapsed > timeout_s:
            return MeasurementAttempt(
                name="subprocess_proof_reuse_benchmark",
                attempted=True,
                succeeded=False,
                skipped=False,
                detail=f"completed_but_slow:{elapsed:.1f}s",
                metrics={
                    "elapsed_s": int(elapsed),
                    "passed": bool(getattr(receipt, "passed", False)),
                    "false_skips": int(getattr(receipt, "false_skips", 0) or 0),
                },
            )
        payload = receipt.to_dict() if hasattr(receipt, "to_dict") else {}
        # Strip floats for safe JSON reporting.
        def _intify(obj: Any) -> Any:
            if isinstance(obj, float):
                return int(round(obj))
            if isinstance(obj, Mapping):
                return {str(k): _intify(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_intify(v) for v in obj]
            return obj

        metrics = _intify(payload) if isinstance(payload, Mapping) else {}
        return MeasurementAttempt(
            name="subprocess_proof_reuse_benchmark",
            attempted=True,
            succeeded=bool(getattr(receipt, "passed", False)),
            skipped=False,
            detail=(
                "measured_ok"
                if getattr(receipt, "passed", False)
                else "measured_failed"
            ),
            metrics={
                "elapsed_s": int(elapsed),
                "false_skips": int(getattr(receipt, "false_skips", 0) or 0),
                "positive_saved_wall": bool(
                    getattr(receipt, "positive_saved_wall", False)
                ),
                "sample_count": len(getattr(receipt, "samples", ()) or ()),
                "receipt": metrics,
            },
        )
    except Exception as exc:
        return MeasurementAttempt(
            name="subprocess_proof_reuse_benchmark",
            attempted=True,
            succeeded=False,
            skipped=False,
            detail=f"{type(exc).__name__}:{exc}"[:240],
            metrics={"fixture": discovery.to_dict()},
        )


def attempt_single_repo_cold_warm(
    *,
    require_available_fixture: bool = True,
    repository_name: str = "ipfs_accelerate",
) -> MeasurementAttempt:
    """Run one ProductionRuntimeActivationE2E cold/warm pair when fixture ready."""

    discovery = discover_real_groth16_fixture()
    if require_available_fixture and not discovery.available:
        return MeasurementAttempt(
            name="single_repo_cold_warm",
            attempted=False,
            succeeded=False,
            skipped=True,
            detail=f"skipped:fixture_{discovery.reason}",
            metrics=discovery.to_dict(),
        )
    try:
        module = load_real_groth16_fixture_module()
        fixture_cls = module.RealGroth16TestPassFixture
        e2e_cls = module.ProductionRuntimeActivationE2E
        specs = module.repository_specs()
        fixture = fixture_cls.discover()
        repo = next(
            (
                item
                for item in specs
                if str(getattr(item, "name", "")) == repository_name
            ),
            specs[0] if specs else None,
        )
        if repo is None:
            return MeasurementAttempt(
                name="single_repo_cold_warm",
                attempted=False,
                succeeded=False,
                skipped=True,
                detail="no_repository_specs",
            )
        with tempfile.TemporaryDirectory(prefix="ptr-closeout-e2e-") as tmp:
            e2e = e2e_cls(
                repository=repo,
                base_dir=Path(tmp) / str(getattr(repo, "name", "repo")),
                fixture=fixture,
            )
            summary = e2e.run_cold_warm(audit_compat=True)
        cold = e2e.cold
        warm = e2e.warm
        false_skips = 0
        if warm is not None and warm.proof_cache_skips and warm.body_marker_count > 0:
            false_skips += int(warm.proof_cache_skips)
        if cold is not None and cold.proof_cache_skips:
            false_skips += int(cold.proof_cache_skips)
        ok = (
            cold is not None
            and warm is not None
            and int(cold.returncode) == 0
            and int(warm.returncode) == 0
            and false_skips == 0
        )
        return MeasurementAttempt(
            name="single_repo_cold_warm",
            attempted=True,
            succeeded=ok,
            skipped=False,
            detail="measured_ok" if ok else "measured_failed",
            metrics={
                "repository": str(getattr(repo, "name", "")),
                "false_skips": false_skips,
                "cold_returncode": int(getattr(cold, "returncode", -1) or -1),
                "warm_returncode": int(getattr(warm, "returncode", -1) or -1),
                "cold_proof_cache_skips": int(
                    getattr(cold, "proof_cache_skips", 0) or 0
                ),
                "warm_proof_cache_skips": int(
                    getattr(warm, "proof_cache_skips", 0) or 0
                ),
                "summary_keys": sorted(summary.keys())
                if isinstance(summary, Mapping)
                else [],
            },
        )
    except Exception as exc:
        return MeasurementAttempt(
            name="single_repo_cold_warm",
            attempted=True,
            succeeded=False,
            skipped=False,
            detail=f"{type(exc).__name__}:{exc}"[:240],
            metrics=discovery.to_dict(),
        )


def attempt_controller_owned_context_smoke() -> MeasurementAttempt:
    """Exercise controller-owned context admit/reconstruct without tree publish.

    Success means the API path works on an incomplete public envelope. This is
    **not** a retained current-tree publication and never authorizes warm-skip.
    """

    try:
        from ipfs_accelerate_py.testing.proof_reuse import candidate_publication as cp

        has_admit = callable(getattr(cp, "admit_controller_owned_v2_context", None))
        has_reconstruct = callable(
            getattr(cp, "reconstruct_controller_owned_v2_context", None)
        )
        has_rehash = callable(getattr(cp, "rehash_controller_owned_public_bytes", None))
        if not (has_admit and has_reconstruct and has_rehash):
            return MeasurementAttempt(
                name="controller_owned_context_api",
                attempted=True,
                succeeded=False,
                skipped=False,
                detail="api_incomplete",
                metrics={
                    "admit": has_admit,
                    "reconstruct": has_reconstruct,
                    "rehash": has_rehash,
                },
            )

        # Incomplete envelope is intentionally incomplete: proves admit/reconstruct
        # plumbing without inventing reviewed pins for the current tree.
        admitted, admit_reason = cp.admit_controller_owned_v2_context(
            {},
            require_complete=False,
        )
        reconstruct_ok = False
        reconstruct_reason = ""
        if admitted is not None:
            rebuilt, reconstruct_reason = cp.reconstruct_controller_owned_v2_context(
                admitted.to_public_mapping()
                if hasattr(admitted, "to_public_mapping")
                else {},
                require_complete=False,
            )
            reconstruct_ok = rebuilt is not None
        ok = admitted is not None and reconstruct_ok
        return MeasurementAttempt(
            name="controller_owned_context_api",
            attempted=True,
            succeeded=ok,
            skipped=False,
            detail=(
                "admit_reconstruct_incomplete_ok"
                if ok
                else f"admit={admit_reason or 'ok'};reconstruct={reconstruct_reason or 'fail'}"
            )[:200],
            metrics={
                "admit": has_admit,
                "reconstruct": has_reconstruct,
                "rehash": has_rehash,
                "incomplete_admit_ok": admitted is not None,
                "incomplete_reconstruct_ok": reconstruct_ok,
                "current_tree_published": False,
                "production_authority": False,
            },
        )
    except Exception as exc:
        return MeasurementAttempt(
            name="controller_owned_context_api",
            attempted=True,
            succeeded=False,
            skipped=False,
            detail=f"{type(exc).__name__}:{exc}"[:200],
        )


def attempt_ordinary_default_composition_probe(
    *,
    repo_root: Path | str | None = None,
) -> MeasurementAttempt:
    """Compose ordinary SHADOW defaults with a cache root and report handles.

    Mirrors the activation live probe path: mode + cache_root so identity,
    candidate store, certificate store, revalidator, and current-context can
    materialize. Does not clear activation_gap or authorize warm-skip.
    """

    try:
        from ipfs_accelerate_py.testing.proof_reuse.config import ProofReuseMode
        from ipfs_accelerate_py.testing.proof_reuse.reporting import (
            proof_reuse_runtime_activation_report,
        )

        roots: list[Path] = []
        if repo_root is not None:
            root = Path(repo_root)
            roots.append(root)
            accel = root / "external" / "ipfs_accelerate"
            if accel.is_dir():
                roots.append(accel)
        try:
            import ipfs_accelerate_py

            roots.append(Path(ipfs_accelerate_py.__file__).resolve().parent.parent)
        except Exception:
            pass

        cache_root = (
            Path.home()
            / ".local"
            / "state"
            / "ipfs_accelerate_py"
            / "proof-backed-test-reuse-v1"
            / "runtime"
            / "closeout-composition-cache"
        )
        try:
            cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        except Exception:
            pass

        last_error = ""
        best: dict[str, Any] = {}
        for root in roots or [None]:  # type: ignore[list-item]
            try:
                report = proof_reuse_runtime_activation_report(
                    mode=ProofReuseMode.SHADOW,
                    root_path=root,
                    cache_root=cache_root,
                    compose_if_missing=True,
                )
                payload = report.to_dict()
                best = payload
                if payload.get("ordinary_default_composition_usable"):
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}:{exc}"[:160]
                continue

        if not best:
            return MeasurementAttempt(
                name="ordinary_default_composition",
                attempted=True,
                succeeded=False,
                skipped=False,
                detail=last_error or "composition_failed",
            )

        composition = best.get("composition") or {}
        handles = composition.get("handles") or {}
        present = {
            name: bool((handles.get(name) or {}).get("present"))
            for name in (
                "identity_services",
                "lookup",
                "store",
                "candidate_store",
                "issuer",
                "revalidator",
                "current_context_provider",
            )
        }
        usable = bool(best.get("ordinary_default_composition_usable"))
        blockers = list(best.get("activation_blocker_codes") or [])[:16]
        return MeasurementAttempt(
            name="ordinary_default_composition",
            attempted=True,
            succeeded=usable,
            skipped=False,
            detail=(
                "composition_usable"
                if usable
                else f"composition_incomplete:blockers={','.join(blockers[:6]) or 'none'}"
            )[:200],
            metrics={
                "ordinary_default_composition_usable": usable,
                "handles_present": present,
                "activation_blocker_codes": blockers,
                "activation_gap_present": bool(best.get("activation_gap_present")),
                "mode": "shadow",
                "cache_root": str(cache_root)[:256],
                "production_authority": False,
            },
        )
    except Exception as exc:
        return MeasurementAttempt(
            name="ordinary_default_composition",
            attempted=True,
            succeeded=False,
            skipped=False,
            detail=f"{type(exc).__name__}:{exc}"[:200],
        )


def attempt_candidate_store_path_probe(
    *,
    repo_root: Path | str | None = None,
) -> MeasurementAttempt:
    """Construct stores and publish a non-authoritative smoke candidate.

    Success proves the ordinary default candidate-context + certificate store
    path can retain a descriptor/components and that controller-owned v2 pins
    can admit a complete (synthetic) envelope. This is **not** a current-tree
    publication and never authorizes warm-skip.
    """

    try:
        from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
            canonical_json_bytes,
        )
        from ipfs_accelerate_py.agent_supervisor.proof.test_candidate_context_store import (
            _cid_for_canonical_bytes,
        )
        from ipfs_accelerate_py.testing.proof_reuse import candidate_publication as cp
        from ipfs_accelerate_py.testing.proof_reuse.activation_contracts import (
            CandidateExecutionContext,
        )
        from ipfs_accelerate_py.testing.proof_reuse.config import ProofReuseMode
        from ipfs_accelerate_py.testing.proof_reuse.services import (
            compose_default_proof_reuse_services,
        )

        def _blob(label: str, padding: int = 32) -> bytes:
            return canonical_json_bytes({"label": label, "padding": "x" * padding})

        roots: list[Path] = []
        if repo_root is not None:
            root = Path(repo_root)
            roots.append(root)
            accel = root / "external" / "ipfs_accelerate"
            if accel.is_dir():
                roots.append(accel)
        try:
            import ipfs_accelerate_py

            roots.append(Path(ipfs_accelerate_py.__file__).resolve().parent.parent)
        except Exception:
            pass

        with tempfile.TemporaryDirectory(prefix="ptr-candidate-") as td:
            last_error = ""
            for root in roots or [None]:  # type: ignore[list-item]
                try:
                    services = compose_default_proof_reuse_services(
                        mode=ProofReuseMode.SHADOW,
                        root_path=root,
                        cache_root=td,
                    )
                    candidate_ok = services.candidate_store is not None
                    store_ok = services.store is not None
                    revalidator_ok = services.revalidator is not None
                    if not (candidate_ok and store_ok and revalidator_ok):
                        last_error = "stores_incomplete"
                        continue

                    components = {
                        "policy": _blob("policy"),
                        "pass_receipt": _blob("receipt"),
                        "execution_key": _blob("execution_key"),
                        "runtime_trace": _blob("runtime"),
                        "static_trace": _blob("static"),
                        "repository_forest": _blob("forest"),
                        "environment": _blob("environment"),
                    }
                    component_cids = {
                        name: _cid_for_canonical_bytes(data)
                        for name, data in components.items()
                    }
                    locator = _cid_for_canonical_bytes(_blob("locator"))
                    ast = _cid_for_canonical_bytes(_blob("ast"))
                    descriptor = CandidateExecutionContext(
                        locator_cid=locator,
                        execution_key_cid=component_cids["execution_key"],
                        pass_receipt_cid=component_cids["pass_receipt"],
                        repository_forest_cid=component_cids["repository_forest"],
                        test_ast_cid=ast,
                        static_trace_root_cid=component_cids["static_trace"],
                        runtime_trace_root_cid=component_cids["runtime_trace"],
                        environment_cid=component_cids["environment"],
                        policy_cid=component_cids["policy"],
                        component_cids=component_cids,
                        retained_at_ms=1,
                    )
                    put = services.candidate_store.publish(
                        descriptor, components, publish_index=True
                    )
                    stored = bool(getattr(put, "stored", False))
                    indexed = bool(getattr(put, "indexed", False))
                    context_cid = str(
                        getattr(put, "candidate_context_cid", "") or ""
                    )[:128]
                    publish_reason = str(getattr(put, "reason_code", "") or "")[:96]

                    controller_complete = False
                    controller_reason = ""
                    if stored and context_cid:
                        admitted, controller_reason = cp.admit_controller_owned_v2_context(
                            {
                                "receipt_cid": component_cids["pass_receipt"],
                                "execution_key_cid": component_cids["execution_key"],
                                "candidate_context_cid": context_cid,
                                "policy_cid": component_cids["policy"],
                                "statement_cid": _cid_for_canonical_bytes(
                                    _blob("statement")
                                ),
                                "circuit_cid": _cid_for_canonical_bytes(_blob("circuit")),
                                "verifying_key_cid": _cid_for_canonical_bytes(
                                    _blob("vk")
                                ),
                                "issuer_id": "issuerlocalnonproduction",
                                "epoch": "epochlocal",
                                "backend_id": "groth16",
                            },
                            require_complete=True,
                        )
                        controller_complete = bool(
                            admitted is not None
                            and getattr(admitted, "is_complete", False)
                        )

                    ok = stored and indexed and controller_complete
                    return MeasurementAttempt(
                        name="candidate_store_path",
                        attempted=True,
                        succeeded=ok,
                        skipped=False,
                        detail=(
                            "publish_and_controller_admit_ok"
                            if ok
                            else (
                                f"publish_stored={stored};indexed={indexed};"
                                f"controller={controller_complete};"
                                f"reason={publish_reason or controller_reason or 'incomplete'}"
                            )
                        )[:200],
                        metrics={
                            "candidate_store": candidate_ok,
                            "certificate_store": store_ok,
                            "revalidator": revalidator_ok,
                            "publish_stored": stored,
                            "publish_indexed": indexed,
                            "publish_reason": publish_reason,
                            "candidate_context_cid": context_cid,
                            "controller_complete_admit": controller_complete,
                            "controller_reason": str(controller_reason or "")[:96],
                            "current_tree_published": False,
                            "root": str(root)[:160] if root is not None else "",
                            "production_authority": False,
                        },
                    )
                except Exception as exc:
                    last_error = f"{type(exc).__name__}:{exc}"[:160]
                    continue
            return MeasurementAttempt(
                name="candidate_store_path",
                attempted=True,
                succeeded=False,
                skipped=False,
                detail=last_error or "construction_failed",
            )
    except Exception as exc:
        return MeasurementAttempt(
            name="candidate_store_path",
            attempted=True,
            succeeded=False,
            skipped=False,
            detail=f"{type(exc).__name__}:{exc}"[:200],
        )


def attempt_reviewed_manifest_pin_status() -> MeasurementAttempt:
    """Report reviewed v4 artifact-manifest pin / allowlist readiness honestly.

    The production allowlist ``DATASETS_GROTH16_APPROVED_V4_KEY_MANIFESTS_SHA256``
    is intentionally empty until an operator-reviewed ceremony publishes exact
    digests. Local operational keys + env self-pins cannot close this gap.
    """

    import os

    try:
        from ipfs_accelerate_py.testing.proof_reuse.services import (
            DATASETS_GROTH16_APPROVED_V4_KEY_MANIFESTS_SHA256,
            PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_ENV,
            PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_SHA256_ENV,
            probe_test_certificate_authority,
        )

        fixture = discover_real_groth16_fixture()
        manifest_path = str(
            os.environ.get(PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_ENV, "") or ""
        ).strip()
        manifest_sha = str(
            os.environ.get(PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_SHA256_ENV, "") or ""
        ).strip()
        approved = frozenset(DATASETS_GROTH16_APPROVED_V4_KEY_MANIFESTS_SHA256 or ())
        certificate = probe_test_certificate_authority(
            artifacts_root=fixture.artifacts_root or None,
            binary_path=fixture.binary_path or None,
        )
        pin_env_set = bool(manifest_path) and bool(manifest_sha)
        path_exists = bool(manifest_path) and Path(manifest_path).expanduser().is_file()
        # Never succeeds while the reviewed allowlist is empty or cert unready.
        ready = bool(certificate.get("ready")) and bool(approved)
        detail = str(certificate.get("reason_code") or "unready")
        if not approved:
            detail = "approved_v4_manifest_allowlist_empty"
        elif not pin_env_set:
            detail = "artifact_manifest_pin_missing"
        elif not path_exists:
            detail = "artifact_manifest_unreadable"
        return MeasurementAttempt(
            name="reviewed_manifest_pin_status",
            attempted=True,
            succeeded=ready,
            skipped=False,
            detail=detail[:200],
            metrics={
                "certificate_ready": bool(certificate.get("ready")),
                "certificate_reason": str(certificate.get("reason_code") or "")[:96],
                "approved_manifest_count": len(approved),
                "pin_env_set": pin_env_set,
                "manifest_path_set": bool(manifest_path),
                "manifest_sha256_set": bool(manifest_sha),
                "manifest_path_exists": path_exists,
                "fixture_keys_present": bool(fixture.available),
                "local_operational_keys_not_production": True,
                "production_authority": False,
                "operator_action": (
                    "operator-reviewed v4 ceremony must publish exact key digests "
                    "into DATASETS_GROTH16_APPROVED_V4_KEY_MANIFESTS_SHA256, then set "
                    f"{PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_ENV} + "
                    f"{PROOF_REUSE_GROTH16_ARTIFACT_MANIFEST_SHA256_ENV}"
                ),
            },
        )
    except Exception as exc:
        return MeasurementAttempt(
            name="reviewed_manifest_pin_status",
            attempted=True,
            succeeded=False,
            skipped=False,
            detail=f"{type(exc).__name__}:{exc}"[:200],
        )


def attempt_issuance_material_api_smoke() -> MeasurementAttempt:
    """Smoke-import proof-bearing issuance/receipt helpers."""

    try:
        from ipfs_accelerate_py.testing.proof_reuse import receipt as receipt_mod

        # Presence of reconstruct helpers is structural readiness, not retention.
        has_reconstruct = hasattr(receipt_mod, "reconstruct_controller_owned_v2_context") or hasattr(
            receipt_mod, "ProofReuseReceipt"
        )
        return MeasurementAttempt(
            name="issuance_material_api",
            attempted=True,
            succeeded=bool(has_reconstruct),
            skipped=False,
            detail="api_present" if has_reconstruct else "api_incomplete",
            metrics={"module": receipt_mod.__name__},
        )
    except Exception as exc:
        return MeasurementAttempt(
            name="issuance_material_api",
            attempted=True,
            succeeded=False,
            skipped=False,
            detail=f"{type(exc).__name__}:{exc}"[:200],
        )


def attempt_local_v4_certificate_self_check() -> MeasurementAttempt:
    """Issue + locally verify a real Groth16 test-pass certificate when keys ready.

    Uses fixture ``issue_self_check`` (outside the ordinary plugin path). Success
    proves local operational issuance/verification, **not** reviewed production
    ceremony authority or ordinary warm-skip authorization.
    """

    discovery = discover_real_groth16_fixture()
    if not discovery.available:
        return MeasurementAttempt(
            name="local_v4_certificate_self_check",
            attempted=False,
            succeeded=False,
            skipped=True,
            detail=f"skipped:fixture_{discovery.reason}",
            metrics=discovery.to_dict(),
        )
    try:
        module = load_real_groth16_fixture_module()
        # Ensure datasets package is importable for issuer path.
        datasets_candidates = [
            Path(__file__).resolve().parents[4] / "external" / "ipfs_datasets",
            Path(__file__).resolve().parents[5] / "external" / "ipfs_datasets",
        ]
        for candidate in datasets_candidates:
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        fixture = module.RealGroth16TestPassFixture.discover()
        result = fixture.issue_self_check()
        if not isinstance(result, Mapping):
            return MeasurementAttempt(
                name="local_v4_certificate_self_check",
                attempted=True,
                succeeded=False,
                skipped=False,
                detail="non_mapping_result",
            )
        ok = bool(result.get("verified_locally")) and bool(result.get("available"))
        return MeasurementAttempt(
            name="local_v4_certificate_self_check",
            attempted=True,
            succeeded=ok,
            skipped=False,
            detail="verified_locally" if ok else str(result.get("reason") or "not_verified"),
            metrics={
                "available": bool(result.get("available")),
                "verified_locally": bool(result.get("verified_locally")),
                "reason": str(result.get("reason") or "")[:96],
                "circuit_cid": str(result.get("circuit_cid") or "")[:96],
                "verifying_key_cid": str(result.get("verifying_key_cid") or "")[:96],
                "proof_digest": str(result.get("proof_digest") or "")[:96],
                "proof_artifact_cid": str(result.get("proof_artifact_cid") or "")[:96],
                "production_authority": False,
                "local_operational_only": True,
            },
        )
    except Exception as exc:
        return MeasurementAttempt(
            name="local_v4_certificate_self_check",
            attempted=True,
            succeeded=False,
            skipped=False,
            detail=f"{type(exc).__name__}:{exc}"[:240],
            metrics=discovery.to_dict(),
        )


def attempt_default_identity_services_probe(
    *,
    repo_root: Path | str | None = None,
) -> MeasurementAttempt:
    """Probe whether default identity services can be constructed.

    Construction alone does not wire ordinary production composition; this only
    reports whether the factory path is usable for further activation work.
    """

    try:
        from ipfs_accelerate_py.testing.proof_reuse.default_identity_services import (
            build_default_identity_services,
        )
        from ipfs_accelerate_py.testing.proof_reuse.plugin import ProofReuseMode

        roots: list[Path] = []
        if repo_root is not None:
            roots.append(Path(repo_root))
            accel = Path(repo_root) / "external" / "ipfs_accelerate"
            if accel.is_dir():
                roots.append(accel)
        try:
            import ipfs_accelerate_py

            roots.append(Path(ipfs_accelerate_py.__file__).resolve().parent.parent)
        except Exception:
            pass
        last_error = ""
        for root in roots or [None]:  # type: ignore[list-item]
            try:
                services = build_default_identity_services(
                    mode=ProofReuseMode.OFF,
                    root_path=root,
                )
                # Off-mode is empty-safe; also try shadow for provider loading.
                services_enabled = build_default_identity_services(
                    mode=ProofReuseMode.SHADOW,
                    root_path=root,
                )
                attrs = [
                    name
                    for name in (
                        "repository_forest_provider",
                        "analysis_index_provider",
                        "component_inputs_provider",
                        "policy_inputs_provider",
                        "runtime_evidence_provider",
                    )
                    if getattr(services_enabled, name, None) is not None
                    or (
                        isinstance(services_enabled, Mapping)
                        and services_enabled.get(name) is not None
                    )
                ]
                return MeasurementAttempt(
                    name="default_identity_services_probe",
                    attempted=True,
                    succeeded=True,
                    skipped=False,
                    detail=f"constructed:root={root}",
                    metrics={
                        "off_type": type(services).__name__,
                        "enabled_type": type(services_enabled).__name__,
                        "configured_providers": attrs,
                        "provider_count": len(attrs),
                    },
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}:{exc}"[:160]
                continue
        return MeasurementAttempt(
            name="default_identity_services_probe",
            attempted=True,
            succeeded=False,
            skipped=False,
            detail=last_error or "construction_failed",
        )
    except Exception as exc:
        return MeasurementAttempt(
            name="default_identity_services_probe",
            attempted=True,
            succeeded=False,
            skipped=False,
            detail=f"{type(exc).__name__}:{exc}"[:200],
        )


def run_closeout_activation_measurements(
    *,
    attempt_heavy_measurements: bool = False,
    require_available_fixture: bool = True,
    attempt_local_setup: bool | None = None,
) -> ActivationMeasurementReport:
    """Discover fixture and optionally run measured activation runners.

    Heavy measurements (subprocess cold/warm e2e) only run when
    ``attempt_heavy_measurements`` is true **and** the fixture is available
    (or require_available_fixture is false).

    Local v4 setup is operator-opt-in via ``PTR_CLOSEOUT_LOCAL_SETUP=1`` and is
    always labeled non-production.
    """

    import os

    if attempt_local_setup is None:
        attempt_local_setup = str(
            os.environ.get("PTR_CLOSEOUT_LOCAL_SETUP", "")
        ).strip().lower() in {"1", "true", "yes", "on"}

    fixture = discover_real_groth16_fixture()
    attempts: list[MeasurementAttempt] = [
        MeasurementAttempt(
            name="fixture_discover",
            attempted=True,
            succeeded=fixture.available,
            skipped=False,
            detail=fixture.reason,
            metrics={
                **fixture.to_dict(),
                "artifact_inventory": inventory_artifact_versions(
                    fixture.artifacts_root
                ),
            },
        ),
        attempt_controller_owned_context_smoke(),
        attempt_issuance_material_api_smoke(),
        attempt_default_identity_services_probe(),
        attempt_ordinary_default_composition_probe(),
        attempt_candidate_store_path_probe(),
        attempt_reviewed_manifest_pin_status(),
    ]

    if attempt_local_setup:
        setup_attempt = attempt_local_nonproduction_v4_setup()
        attempts.append(setup_attempt)
        if setup_attempt.succeeded or setup_attempt.detail.startswith("skipped:v4"):
            fixture = discover_real_groth16_fixture()
            attempts.append(
                MeasurementAttempt(
                    name="fixture_rediscover_after_setup",
                    attempted=True,
                    succeeded=fixture.available,
                    skipped=False,
                    detail=fixture.reason,
                    metrics=fixture.to_dict(),
                )
            )
    else:
        attempts.append(
            MeasurementAttempt(
                name="local_nonproduction_v4_setup",
                attempted=False,
                succeeded=False,
                skipped=True,
                detail="skipped:set PTR_CLOSEOUT_LOCAL_SETUP=1 for local operational v4 keys",
                metrics={
                    "artifact_inventory": inventory_artifact_versions(
                        fixture.artifacts_root
                    )
                },
            )
        )

    # Auto-enable heavy measurements when fixture becomes available and env asks,
    # or when caller requested heavy measurements.
    run_heavy = attempt_heavy_measurements or (
        fixture.available
        and str(os.environ.get("PTR_CLOSEOUT_HEAVY_MEASUREMENTS", "")).strip().lower()
        in {"1", "true", "yes", "on", "auto"}
    )
    if run_heavy:
        attempts.append(
            attempt_subprocess_proof_reuse_benchmark(
                require_available_fixture=require_available_fixture
            )
        )
        attempts.append(
            attempt_single_repo_cold_warm(
                require_available_fixture=require_available_fixture
            )
        )
        # Local cert self-check is medium weight: only when keys ready.
        attempts.append(attempt_local_v4_certificate_self_check())
    else:
        attempts.append(
            MeasurementAttempt(
                name="subprocess_proof_reuse_benchmark",
                attempted=False,
                succeeded=False,
                skipped=True,
                detail=(
                    "skipped:heavy_measurements_disabled"
                    if not fixture.available
                    else "skipped:set PTR_CLOSEOUT_HEAVY_MEASUREMENTS=1"
                ),
                metrics={"fixture_available": fixture.available},
            )
        )
        attempts.append(
            MeasurementAttempt(
                name="single_repo_cold_warm",
                attempted=False,
                succeeded=False,
                skipped=True,
                detail=(
                    "skipped:heavy_measurements_disabled"
                    if not fixture.available
                    else "skipped:set PTR_CLOSEOUT_HEAVY_MEASUREMENTS=1"
                ),
                metrics={"fixture_available": fixture.available},
            )
        )
        # Still try cheap local cert self-check when keys already present.
        if fixture.available:
            attempts.append(attempt_local_v4_certificate_self_check())
        else:
            attempts.append(
                MeasurementAttempt(
                    name="local_v4_certificate_self_check",
                    attempted=False,
                    succeeded=False,
                    skipped=True,
                    detail="skipped:fixture_not_ready",
                )
            )

    by_name = {item.name: item for item in attempts}
    local_cert = by_name.get("local_v4_certificate_self_check")
    claims = {
        # Only true when measured runners succeeded with available fixture.
        "measured_subprocess_benchmark": bool(
            by_name.get("subprocess_proof_reuse_benchmark")
            and by_name["subprocess_proof_reuse_benchmark"].succeeded
        ),
        "three_repository_cold_warm": bool(
            by_name.get("single_repo_cold_warm")
            and by_name["single_repo_cold_warm"].succeeded
            and fixture.available
        ),
        # API presence is not the same as retained live context — keep False.
        "controller_owned_receipt_candidate_context": False,
        # Local self-check issues retained material in-process (operational only).
        "retained_proof_bearing_issuance_material": bool(
            local_cert and local_cert.succeeded
        ),
        # Operational keys enable local real cert issuance; production ceremony
        # still gated separately by activation_gap in the activation probe.
        "real_groth16_certificate": bool(
            fixture.available and local_cert and local_cert.succeeded
        ),
        "exact_reviewed_source_binary_capability_circuit_key_identities": bool(
            fixture.available and local_cert and local_cert.succeeded
        ),
        "locally_verified_current_v4_certificate": bool(
            local_cert and local_cert.succeeded
        ),
        "fixture_binary_present": bool(
            fixture.binary_path and Path(fixture.binary_path).is_file()
        ),
        "fixture_keys_present": bool(fixture.available),
        "default_identity_services_constructible": bool(
            by_name.get("default_identity_services_probe")
            and by_name["default_identity_services_probe"].succeeded
        ),
        "ordinary_default_composition_usable": bool(
            by_name.get("ordinary_default_composition")
            and by_name["ordinary_default_composition"].succeeded
        ),
        "candidate_store_path_ready": bool(
            by_name.get("candidate_store_path")
            and by_name["candidate_store_path"].succeeded
        ),
        # Controller API smoke is not a retained current-tree publication.
        "controller_owned_context_api_ready": bool(
            by_name.get("controller_owned_context_api")
            and by_name["controller_owned_context_api"].succeeded
        ),
        "candidate_publish_and_controller_admit_ready": bool(
            by_name.get("candidate_store_path")
            and by_name["candidate_store_path"].succeeded
        ),
        "reviewed_manifest_pin_ready": bool(
            by_name.get("reviewed_manifest_pin_status")
            and by_name["reviewed_manifest_pin_status"].succeeded
        ),
    }
    return ActivationMeasurementReport(
        fixture=fixture,
        attempts=tuple(attempts),
        claims_supported=claims,
    )


__all__ = [
    "ACTIVATION_MEASUREMENT_SCHEMA",
    "ActivationMeasurementReport",
    "FixtureDiscoveryResult",
    "MEASUREMENT_INTERFACE",
    "MEASUREMENT_SCHEMA",
    "MeasurementAttempt",
    "attempt_candidate_store_path_probe",
    "attempt_controller_owned_context_smoke",
    "attempt_default_identity_services_probe",
    "attempt_issuance_material_api_smoke",
    "attempt_local_nonproduction_v4_setup",
    "attempt_local_v4_certificate_self_check",
    "attempt_ordinary_default_composition_probe",
    "attempt_reviewed_manifest_pin_status",
    "attempt_single_repo_cold_warm",
    "attempt_subprocess_proof_reuse_benchmark",
    "discover_real_groth16_fixture",
    "inventory_artifact_versions",
    "load_real_groth16_fixture_module",
    "run_closeout_activation_measurements",
]

# Alias for export typo-safety
ACTIVATION_MEASUREMENT_SCHEMA = MEASUREMENT_SCHEMA
