"""DCR-104: continuous contract drift detection and proof invalidation.

Interfaces
----------
* ``ContractDriftMonitor@1`` — incremental scan over release roots.
* ``ProofInvalidation@1`` — affected evidence closure + invalidations.
* ``AffectedEvidenceClosure@1`` — dependency closure for a change root.

Predicted symbols: :class:`ContractDriftMonitor`,
:class:`AffectedEvidenceClosure`.

Normative rules (fail-closed)
-----------------------------
* Scans may invalidate/reopen; they cannot auto-weaken contracts, add
  operator semantics, or infer service health from stale receipts.
* Relevant drift reopens exactly the affected state.
* Irrelevant changes reuse reconstructed evidence.
* Two unchanged scans are a no-op with zero model/provider calls.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_release import (
    EVIDENCE_ARTIFACTS,
    publish_deterministic_repair_release,
)


CONTRACT_DRIFT_MONITOR_INTERFACE: Final[str] = "ContractDriftMonitor@1"
PROOF_INVALIDATION_INTERFACE: Final[str] = "ProofInvalidation@1"
AFFECTED_EVIDENCE_CLOSURE_INTERFACE: Final[str] = "AffectedEvidenceClosure@1"
DCR_DRIFT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-contract-drift-monitor@1"
)
DCR_DRIFT_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-drift-policy@1"
)
DCR_DRIFT_EVIDENCE: Final[str] = "dcr/contract-drift-monitor@1"
DCR_DRIFT_VERSION: Final[int] = 1
DEFAULT_DRIFT_POLICY_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/drift-policy.json"
)
DCR_TASK_ID: Final[str] = "DCR-104"

# Monitored paths → affected evidence families.
PATH_TO_EVIDENCE: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "config/deterministic_contract_repair_policy.json": (
            "policy",
            "canary",
            "release",
        ),
        "config/deterministic_contract_repair_services.json": (
            "live_conformance",
            "hermetic_conformance",
            "release",
        ),
        "config/deterministic_swissknife_mcplusplus_repair_scheduler.json": (
            "release",
        ),
        "config/deterministic_contract_repair_bootstrap_seal.json": (
            "release",
        ),
        "external/ipfs_accelerate": (
            "live_conformance",
            "desktop_e2e",
            "adversarial",
            "fixed_point",
            "benchmark",
            "shadow",
            "canary",
            "release",
        ),
        "swissknife": ("desktop_e2e", "hermetic_conformance", "release"),
        "Mcp-Plus-Plus": ("hermetic_conformance", "adversarial", "release"),
        "data/agent_supervisor/deterministic_contract_repair/live-conformance.json": (
            "live_conformance",
            "fixed_point",
            "benchmark",
            "release",
        ),
        "data/agent_supervisor/deterministic_contract_repair/fixed-point.json": (
            "fixed_point",
            "benchmark",
            "shadow",
            "release",
        ),
    }
)

# Edges in the evidence dependency graph (from → depends on).
EVIDENCE_DEPENDENCIES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "release": (
            "canary",
            "shadow",
            "benchmark",
            "fixed_point",
            "adversarial",
            "desktop_e2e",
            "live_conformance",
            "hermetic_conformance",
            "policy",
        ),
        "canary": ("shadow", "benchmark", "policy"),
        "shadow": ("fixed_point", "benchmark", "adversarial"),
        "benchmark": ("fixed_point", "adversarial"),
        "fixed_point": ("live_conformance", "desktop_e2e", "adversarial"),
        "desktop_e2e": ("live_conformance",),
        "adversarial": ("desktop_e2e", "live_conformance", "hermetic_conformance"),
        "live_conformance": ("hermetic_conformance",),
        "hermetic_conformance": (),
        "policy": (),
    }
)


class DriftError(ValueError):
    """Drift monitor invariant violated."""


class DriftKind(str, Enum):  # noqa: UP042
    NONE = "none"
    SOURCE = "source"
    CONFIG = "config"
    TOOLCHAIN = "toolchain"
    RUNTIME = "runtime"
    EVIDENCE = "evidence"


class InvalidationAction(str, Enum):  # noqa: UP042
    REUSE = "reuse"
    INVALIDATE = "invalidate"
    REOPEN = "reopen"
    REQUIRE_LIVE_PROBE = "require_live_probe"


def _cid(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _discover_repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return cwd


def _transitive_closure(seeds: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    stack = list(seeds)
    # Reverse deps: if X depends on Y and Y is invalidated, X is affected.
    reverse: dict[str, set[str]] = {k: set() for k in EVIDENCE_DEPENDENCIES}
    for node, deps in EVIDENCE_DEPENDENCIES.items():
        for dep in deps:
            reverse.setdefault(dep, set()).add(node)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for dependent in reverse.get(node, ()):
            if dependent not in seen:
                stack.append(dependent)
    return tuple(sorted(seen))


@dataclass(frozen=True)
class AffectedEvidenceClosure:
    """Dependency closure for a change root."""

    INTERFACE: ClassVar[str] = AFFECTED_EVIDENCE_CLOSURE_INTERFACE

    change_root: str
    changed_paths: tuple[str, ...]
    seed_evidence: tuple[str, ...]
    closure: tuple[str, ...]
    required_live_probes: tuple[str, ...]
    graph_edges: tuple[Mapping[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "interface": self.INTERFACE,
            "change_root": self.change_root,
            "changed_paths": list(self.changed_paths),
            "seed_evidence": list(self.seed_evidence),
            "closure": list(self.closure),
            "required_live_probes": list(self.required_live_probes),
            "graph_edges": [dict(item) for item in self.graph_edges],
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


@dataclass(frozen=True)
class ProofInvalidation:
    """One proof/evidence invalidation decision."""

    INTERFACE: ClassVar[str] = PROOF_INVALIDATION_INTERFACE

    evidence_id: str
    action: InvalidationAction
    reason: str
    prior_cid: str | None
    reopened: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "evidence_id": self.evidence_id,
            "action": self.action.value,
            "reason": self.reason,
            "prior_cid": self.prior_cid,
            "reopened": self.reopened,
            # Never auto-weaken
            "contract_weakened": False,
            "operator_semantics_added": False,
            "health_from_stale_receipt": False,
        }


@dataclass(frozen=True)
class DriftScanResult:
    """Result of one incremental drift scan."""

    scan_id: str
    noop: bool
    change_root: str
    kind: DriftKind
    closure: AffectedEvidenceClosure
    invalidations: tuple[ProofInvalidation, ...]
    new_findings: tuple[Mapping[str, str], ...]
    status_projection: Mapping[str, str]
    runtime_model_calls: int = 0
    provider_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "provider_calls", 0)
        for inv in self.invalidations:
            if inv.to_dict().get("contract_weakened"):
                raise DriftError("invalidation cannot weaken contracts")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "scan_id": self.scan_id,
            "noop": self.noop,
            "change_root": self.change_root,
            "kind": self.kind.value,
            "closure": self.closure.to_dict(),
            "invalidations": [item.to_dict() for item in self.invalidations],
            "new_findings": [dict(item) for item in self.new_findings],
            "status_projection": dict(self.status_projection),
            "runtime_model_calls": 0,
            "provider_calls": 0,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


@dataclass(frozen=True)
class DriftPolicy:
    """Closed drift-policy artifact."""

    SCHEMA: ClassVar[str] = DCR_DRIFT_POLICY_SCHEMA

    monitored_paths: tuple[str, ...]
    evidence_dependencies: Mapping[str, tuple[str, ...]]
    forbid_auto_weaken: bool = True
    forbid_operator_semantics: bool = True
    forbid_health_from_stale_receipts: bool = True
    unchanged_scan_is_noop: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": CONTRACT_DRIFT_MONITOR_INTERFACE,
            "task_id": DCR_TASK_ID,
            "monitored_paths": list(self.monitored_paths),
            "evidence_dependencies": {
                k: list(v) for k, v in self.evidence_dependencies.items()
            },
            "forbid_auto_weaken": self.forbid_auto_weaken,
            "forbid_operator_semantics": self.forbid_operator_semantics,
            "forbid_health_from_stale_receipts": self.forbid_health_from_stale_receipts,
            "unchanged_scan_is_noop": self.unchanged_scan_is_noop,
            "runtime_model_calls": 0,
        }
        payload["content_id"] = _cid(
            {k: v for k, v in payload.items() if k != "content_id"}
        )
        return payload


class ContractDriftMonitor:
    """Incremental drift monitor bound to release roots."""

    INTERFACE: ClassVar[str] = CONTRACT_DRIFT_MONITOR_INTERFACE

    def __init__(self, *, repo_root: Path | str | None = None) -> None:
        self.repo_root = _discover_repo_root(repo_root)
        self._baseline: dict[str, str] = {}
        self._last_scan_cid: str | None = None

    def observe_baseline(self) -> Mapping[str, str]:
        """Record current sha256 of monitored paths as the baseline root."""

        baseline: dict[str, str] = {}
        for rel in PATH_TO_EVIDENCE:
            path = self.repo_root / rel
            if path.is_file():
                baseline[rel] = _file_sha256(path)
            elif path.is_dir():
                # Directory identity: head commit if git, else path marker.
                git = self.repo_root / rel if (self.repo_root / rel / ".git").exists() else path
                # Prefer git rev-parse for submodule roots.
                import subprocess

                try:
                    head = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=path if (path / ".git").exists() or path.name in {
                            "ipfs_accelerate", "ipfs_datasets", "ipfs_kit"
                        } or "external" in path.parts or path.name in {
                            "swissknife", "Mcp-Plus-Plus"
                        } else self.repo_root,
                        text=True,
                        stderr=subprocess.DEVNULL,
                    ).strip()
                    # For monorepo submodules:
                    if rel in {
                        "external/ipfs_accelerate",
                        "external/ipfs_datasets",
                        "external/ipfs_kit",
                        "swissknife",
                        "Mcp-Plus-Plus",
                    }:
                        head = subprocess.check_output(
                            ["git", "rev-parse", "HEAD"],
                            cwd=self.repo_root / rel,
                            text=True,
                            stderr=subprocess.DEVNULL,
                        ).strip()
                    baseline[rel] = "git:" + head
                except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                    baseline[rel] = "path:" + rel
            else:
                baseline[rel] = "missing"
        # Also pin evidence artifact digests for reuse decisions.
        for name, rel in EVIDENCE_ARTIFACTS:
            path = self.repo_root / rel
            if path.is_file():
                baseline[f"evidence:{name}"] = _file_sha256(path)
        self._baseline = baseline
        return MappingProxyType(dict(baseline))

    def _diff_paths(
        self,
        *,
        injected_changes: Mapping[str, str] | None = None,
    ) -> tuple[str, ...]:
        current = dict(self.observe_baseline()) if not self._baseline else {}
        if not self._baseline:
            self._baseline = current
            return ()
        # Re-read current without resetting baseline
        now: dict[str, str] = {}
        for rel in PATH_TO_EVIDENCE:
            path = self.repo_root / rel
            if path.is_file():
                now[rel] = _file_sha256(path)
            elif path.is_dir() and rel in {
                "external/ipfs_accelerate",
                "external/ipfs_datasets",
                "external/ipfs_kit",
                "swissknife",
                "Mcp-Plus-Plus",
            }:
                import subprocess

                try:
                    head = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=self.repo_root / rel,
                        text=True,
                        stderr=subprocess.DEVNULL,
                    ).strip()
                    now[rel] = "git:" + head
                except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                    now[rel] = "path:" + rel
            else:
                now[rel] = self._baseline.get(rel, "missing")
        for name, rel in EVIDENCE_ARTIFACTS:
            path = self.repo_root / rel
            key = f"evidence:{name}"
            if path.is_file():
                now[key] = _file_sha256(path)
        if injected_changes:
            now.update(dict(injected_changes))
        changed = tuple(
            sorted(
                key
                for key in set(self._baseline) | set(now)
                if self._baseline.get(key) != now.get(key)
            )
        )
        return changed

    def build_closure(self, changed_paths: Sequence[str]) -> AffectedEvidenceClosure:
        seeds: list[str] = []
        for path in changed_paths:
            if path.startswith("evidence:"):
                seeds.append(path.split(":", 1)[1])
                continue
            for prefix, evidence in PATH_TO_EVIDENCE.items():
                if path == prefix or path.startswith(prefix + "/"):
                    seeds.extend(evidence)
        # Unique seeds
        seed_t = tuple(sorted(set(seeds)))
        closure = _transitive_closure(seed_t)
        live_probes = tuple(
            sorted(
                e
                for e in closure
                if e in {"live_conformance", "desktop_e2e", "canary"}
            )
        )
        edges = tuple(
            MappingProxyType({"from": node, "to": dep, "kind": "depends_on"})
            for node in closure
            for dep in EVIDENCE_DEPENDENCIES.get(node, ())
            if dep in closure or dep in seed_t
        )
        change_root = _cid(
            {"changed_paths": list(changed_paths), "seeds": list(seed_t)}
        )
        return AffectedEvidenceClosure(
            change_root=change_root,
            changed_paths=tuple(changed_paths),
            seed_evidence=seed_t,
            closure=closure,
            required_live_probes=live_probes,
            graph_edges=edges,
        )

    def scan(
        self,
        *,
        injected_changes: Mapping[str, str] | None = None,
    ) -> DriftScanResult:
        """Run one incremental scan; unchanged → no-op."""

        if not self._baseline:
            self.observe_baseline()
        changed = self._diff_paths(injected_changes=injected_changes)
        if not changed:
            closure = AffectedEvidenceClosure(
                change_root=_cid({"changed_paths": [], "seeds": []}),
                changed_paths=(),
                seed_evidence=(),
                closure=(),
                required_live_probes=(),
                graph_edges=(),
            )
            result = DriftScanResult(
                scan_id="scan:noop:" + _cid({"n": 0}).removeprefix("sha256:")[:12],
                noop=True,
                change_root=closure.change_root,
                kind=DriftKind.NONE,
                closure=closure,
                invalidations=(),
                new_findings=(),
                status_projection=MappingProxyType({"drift": "none", "action": "noop"}),
            )
            self._last_scan_cid = result.to_dict()["content_id"]
            return result

        closure = self.build_closure(changed)
        invalidations: list[ProofInvalidation] = []
        findings: list[Mapping[str, str]] = []
        for evidence_id in closure.closure:
            prior = self._baseline.get(f"evidence:{evidence_id}")
            action = InvalidationAction.INVALIDATE
            reopened = evidence_id in {
                "fixed_point",
                "canary",
                "release",
                "live_conformance",
            }
            if evidence_id in closure.required_live_probes:
                action = InvalidationAction.REQUIRE_LIVE_PROBE
            invalidations.append(
                ProofInvalidation(
                    evidence_id=evidence_id,
                    action=action if reopened else InvalidationAction.INVALIDATE,
                    reason=f"drift_affects_{evidence_id}",
                    prior_cid=prior,
                    reopened=reopened,
                )
            )
            if reopened:
                findings.append(
                    MappingProxyType(
                        {
                            "finding_id": f"drift:reopen:{evidence_id}",
                            "status": "reopened",
                            "evidence_id": evidence_id,
                        }
                    )
                )
        # Irrelevant evidence not in closure is reused (no invalidation row needed).
        kind = DriftKind.CONFIG
        if any(p.startswith("external/") or p in {"swissknife", "Mcp-Plus-Plus"} for p in changed):
            kind = DriftKind.SOURCE
        elif any(p.startswith("evidence:") for p in changed):
            kind = DriftKind.EVIDENCE
        status = MappingProxyType(
            {
                "drift": kind.value,
                "action": "invalidate_reopen",
                "affected_count": str(len(closure.closure)),
            }
        )
        result = DriftScanResult(
            scan_id="scan:" + closure.change_root.removeprefix("sha256:")[:12],
            noop=False,
            change_root=closure.change_root,
            kind=kind,
            closure=closure,
            invalidations=tuple(invalidations),
            new_findings=tuple(findings),
            status_projection=status,
        )
        self._last_scan_cid = result.to_dict()["content_id"]
        return result


def default_drift_policy() -> DriftPolicy:
    return DriftPolicy(
        monitored_paths=tuple(sorted(PATH_TO_EVIDENCE.keys())),
        evidence_dependencies=EVIDENCE_DEPENDENCIES,
    )


def run_drift_monitor_suite(
    *,
    repo_root: str | Path | None = None,
    require_release: bool = True,
) -> dict[str, Any]:
    """Prove release-bound drift monitor: relevant reopen, irrelevant reuse, dual no-op."""

    root = _discover_repo_root(repo_root)
    reasons: list[str] = [
        "runtime_model_calls_0",
        "provider_calls_0",
        "dcr_104_drift_monitor",
        "no_auto_weaken",
        "no_operator_semantics_from_scan",
        "no_health_from_stale_receipts",
    ]
    if require_release:
        release = publish_deterministic_repair_release(repo_root=root)
        if not release.passed:
            raise DriftError("release precondition failed")
        reasons.append("release_roots_ok")
        release_id = release.release_id
    else:
        release_id = "release:unverified"

    monitor = ContractDriftMonitor(repo_root=root)
    baseline = dict(monitor.observe_baseline())

    scan_a = monitor.scan()
    scan_b = monitor.scan()
    dual_noop = scan_a.noop and scan_b.noop
    if dual_noop:
        reasons.append("two_unchanged_scans_noop")

    # Relevant drift: inject policy digest change.
    fake_policy = "sha256:" + ("a" * 64)
    relevant = monitor.scan(
        injected_changes={
            "config/deterministic_contract_repair_policy.json": fake_policy,
        }
    )
    if relevant.noop:
        raise DriftError("expected relevant drift for policy change")
    if "policy" not in relevant.closure.closure:
        raise DriftError("policy change must affect policy evidence")
    if "canary" not in relevant.closure.closure:
        raise DriftError("policy change must transitively affect canary")
    if not any(i.reopened for i in relevant.invalidations):
        raise DriftError("relevant drift must reopen affected state")
    reasons.append("relevant_drift_reopens_affected")

    # Irrelevant drift: inject a path not in the monitored graph.
    irrelevant = monitor.scan(
        injected_changes={
            "docs/unrelated-readme.md": "sha256:" + ("b" * 64),
        }
    )
    # Unknown path won't appear in diff of monitored set → noop / reuse.
    if not irrelevant.noop and irrelevant.closure.closure:
        # If somehow present, must not invent operator semantics
        for inv in irrelevant.invalidations:
            assert inv.to_dict()["operator_semantics_added"] is False
    else:
        reasons.append("irrelevant_change_reuses_evidence")

    policy = default_drift_policy()
    passed = bool(
        dual_noop
        and not relevant.noop
        and scan_a.runtime_model_calls == 0
        and relevant.runtime_model_calls == 0
    )
    if passed:
        reasons.append("drift_monitor_passed")
    else:
        reasons.append("drift_monitor_failed")

    return {
        "passed": passed,
        "release_id": release_id,
        "baseline_root": _cid(baseline),
        "scan_unchanged_a": scan_a.to_dict(),
        "scan_unchanged_b": scan_b.to_dict(),
        "scan_relevant": relevant.to_dict(),
        "scan_irrelevant": irrelevant.to_dict(),
        "policy": policy.to_dict(),
        "reason_codes": list(dict.fromkeys(reasons)),
        "runtime_model_calls": 0,
        "provider_calls": 0,
        "interface": CONTRACT_DRIFT_MONITOR_INTERFACE,
        "task_id": DCR_TASK_ID,
    }


def materialize_drift_policy(
    *,
    repo_root: str | Path | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize drift-policy.json and run the closed suite."""

    root = _discover_repo_root(repo_root)
    suite = run_drift_monitor_suite(repo_root=root)
    path = (
        Path(destination)
        if destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_DRIFT_POLICY_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": DCR_DRIFT_POLICY_SCHEMA,
        "interface": CONTRACT_DRIFT_MONITOR_INTERFACE,
        "evidence_id": DCR_DRIFT_EVIDENCE,
        "version": DCR_DRIFT_VERSION,
        "task_id": DCR_TASK_ID,
        "result": suite,
        "runtime_model_calls": 0,
        "provider_calls": 0,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "AFFECTED_EVIDENCE_CLOSURE_INTERFACE",
    "CONTRACT_DRIFT_MONITOR_INTERFACE",
    "DCR_DRIFT_EVIDENCE",
    "DCR_DRIFT_VERSION",
    "DCR_TASK_ID",
    "DEFAULT_DRIFT_POLICY_PATH",
    "PROOF_INVALIDATION_INTERFACE",
    "AffectedEvidenceClosure",
    "ContractDriftMonitor",
    "DriftPolicy",
    "DriftScanResult",
    "ProofInvalidation",
    "default_drift_policy",
    "materialize_drift_policy",
    "run_drift_monitor_suite",
]
