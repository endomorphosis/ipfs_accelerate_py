"""Thin IPFS Kit VFS symbolic-assurance job adapter.

Assembles a locked, content-identified profile from
``config/ipfs_kit_vfs_symbolic_assurance.json`` and delegates to the generic
assurance engines.  Optional providers are never imported at module import
time; adapter callables are resolved lazily from a closed registry.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.control.symbolic_assurance_rollout import (
    AdversarialGateReport,
    AdversarialInjection,
    AssuranceRolloutBinding,
    AssuranceRolloutDecision,
    AssuranceRolloutMode,
    AssuranceRolloutPolicy,
    AssuranceRolloutProfile,
    AssuranceRolloutSchemas,
    GateDefinition,
    SymbolicAssurancePublicAPI,
    SymbolicAssuranceRolloutError,
    build_default_rollout_binding,
    build_default_rollout_policy,
    build_frozen_adversarial_population,
    evaluate_adversarial_gates,
    evaluate_symbolic_assurance_rollout,
    freeze_multi_repository_fixture,
    project_bounded_findings,
    project_bounded_receipts,
    project_bounded_status,
    run_symbolic_assurance_e2e,
    verify_adversarial_e2e_report,
    verify_symbolic_assurance_rollout,
)

CONFIG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ipfs-kit-vfs-symbolic-assurance-config@1"
)
DEFAULT_CONFIG_RELATIVE: Final = "config/ipfs_kit_vfs_symbolic_assurance.json"
CLOSED_ADAPTERS: Final[frozenset[str]] = frozenset(
    {
        "inventory",
        "contracts",
        "differential",
        "parity",
        "benchmark",
        "pilot",
        "rollout",
        "verify",
    }
)
_OPTIONAL_PROVIDER_ROOTS: Final[tuple[str, ...]] = (
    "openai",
    "anthropic",
    "groq",
    "litellm",
    "google.generativeai",
    "torch",
    "transformers",
    "neo4j",
    "duckdb",
)

# Module-level state remains empty at cold import.
_LOADED_OPTIONAL_PROVIDERS: set[str] = set()
_CONFIG_CACHE: dict[str, "IpfsKitVfsAssuranceConfig"] = {}


class IpfsKitVfsAssuranceError(ValueError):
    """Profile, config, or adapter registry failure."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _repo_root() -> Path:
    # integrations/ipfs_kit_vfs_assurance.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def _is_safe_relative(root: str) -> bool:
    if not isinstance(root, str) or not root or root != root.strip():
        return False
    if root.startswith("/") or root.startswith("\\"):
        return False
    path = Path(root)
    if path.is_absolute():
        return False
    parts = path.parts
    if any(part in {"..", ""} for part in parts):
        return False
    return True


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    module: str
    factory: str
    lazy: bool = True
    optional_provider: bool = False

    def __post_init__(self) -> None:
        if self.name not in CLOSED_ADAPTERS:
            raise IpfsKitVfsAssuranceError(f"adapter {self.name!r} is not in closed registry")
        if not self.lazy:
            raise IpfsKitVfsAssuranceError("adapters must be lazy")
        if self.optional_provider:
            raise IpfsKitVfsAssuranceError(
                "optional providers must not register as default adapters"
            )
        if not self.module or not self.factory:
            raise IpfsKitVfsAssuranceError("adapter module and factory are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "factory": self.factory,
            "lazy": True,
            "optional_provider": False,
        }


@dataclass(frozen=True)
class IpfsKitVfsAssuranceConfig:
    """Immutable, bounded, content-identified job configuration."""

    raw: Mapping[str, Any]
    content_id: str
    path: str
    profile: AssuranceRolloutProfile
    adapters: Mapping[str, AdapterSpec]
    safe_relative_roots: tuple[str, ...]
    operation_invariant_error_mappings: Mapping[str, Any]
    cli_subcommands: tuple[str, ...]
    exit_codes: Mapping[str, int]
    schemas_extra: Mapping[str, Any]
    authority_flags: Mapping[str, bool]

    @property
    def binding_id(self) -> str:
        return str(self.raw["binding_id"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_id": self.content_id,
            "path": self.path,
            "binding_id": self.binding_id,
            "profile_id": self.profile.profile_id,
            "profile_content_id": self.profile.profile_content_id,
            "adapters": {
                name: spec.to_dict() for name, spec in sorted(self.adapters.items())
            },
            "safe_relative_roots": list(self.safe_relative_roots),
            "cli_subcommands": list(self.cli_subcommands),
            "exit_codes": dict(self.exit_codes),
            "authority_flags": dict(self.authority_flags),
            "automatic_mutation_enabled": False,
        }


def default_config_path(*, checkout_root: Path | None = None) -> Path:
    root = checkout_root or _repo_root()
    return (root / DEFAULT_CONFIG_RELATIVE).resolve()


def load_assurance_config(
    path: Path | str | None = None,
    *,
    checkout_root: Path | None = None,
) -> IpfsKitVfsAssuranceConfig:
    """Load and validate the immutable job configuration."""

    config_path = Path(path) if path is not None else default_config_path(
        checkout_root=checkout_root
    )
    key = str(config_path)
    cached = _CONFIG_CACHE.get(key)
    if cached is not None:
        return cached

    if not config_path.is_file():
        raise IpfsKitVfsAssuranceError(f"config not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IpfsKitVfsAssuranceError(f"config is not valid JSON: {config_path}") from exc
    if not isinstance(raw, dict):
        raise IpfsKitVfsAssuranceError("config root must be an object")
    if raw.get("schema") != CONFIG_SCHEMA:
        raise IpfsKitVfsAssuranceError("unsupported config schema")
    if raw.get("automatic_mutation_enabled") is True:
        raise IpfsKitVfsAssuranceError("config cannot enable automatic mutation")
    if int(raw.get("config_version", 0)) != 1:
        raise IpfsKitVfsAssuranceError("unsupported config_version")

    # Verify content identity over the document without content_id.
    material = {k: v for k, v in raw.items() if k != "content_id"}
    expected = _content_id(material)
    declared = raw.get("content_id")
    if declared is not None and declared != expected:
        raise IpfsKitVfsAssuranceError("config content_id mismatch")

    identity = dict(raw.get("identity") or {})
    schemas_raw = dict(raw.get("schemas") or {})
    schemas = AssuranceRolloutSchemas(
        adversarial_e2e_gate=schemas_raw.get(
            "adversarial_e2e_gate", "vfs/adversarial-e2e-gate@1"
        ),
        shadow_rollout_report=schemas_raw.get(
            "shadow_rollout_report", "vfs/shadow-rollout-report@1"
        ),
        rollout_decision=schemas_raw.get(
            "rollout_decision", "vfs/symbolic-rollout-decision@1"
        ),
        control_request=schemas_raw.get(
            "control_request", "vfs/symbolic-control-request@1"
        ),
        control_result=schemas_raw.get(
            "control_result", "vfs/symbolic-control-result@1"
        ),
        bounded_status=schemas_raw.get(
            "bounded_status", "vfs/symbolic-bounded-status@1"
        ),
        bounded_findings=schemas_raw.get(
            "bounded_findings", "vfs/symbolic-bounded-findings@1"
        ),
        bounded_receipts=schemas_raw.get(
            "bounded_receipts", "vfs/symbolic-bounded-receipts@1"
        ),
        public_api=schemas_raw.get("public_api", "vfs/symbolic-public-api@1"),
        version=int(schemas_raw.get("version", 1)),
    )
    gates = tuple(
        GateDefinition.from_dict(item) for item in raw.get("gates", ())
    )
    authority = {
        str(k): bool(v) for k, v in dict(raw.get("authority_flags") or {}).items()
    }
    if any(authority.values()):
        raise IpfsKitVfsAssuranceError("authority flags must remain non-authoritative")

    profile = AssuranceRolloutProfile(
        profile_id=str(raw["profile_id"]),
        behavior_id=str(raw["behavior_id"]),
        objective_id=str(raw["objective_id"]),
        objective_revision=str(raw["objective_revision"]),
        requirement_id=str(raw["requirement_id"]),
        gates=gates,
        schemas=schemas,
        default_exclusion_prefixes=tuple(raw.get("default_exclusion_prefixes") or ()),
        default_fixture_repositories=dict(
            raw.get("default_fixture_repositories") or {}
        ),
        default_fixture_id=str(
            identity.get("default_fixture_id", "fixture:vfs-adversarial-e2e@1")
        ),
        default_fixture_revision=str(
            identity.get("default_fixture_revision", "fixture-revision:1")
        ),
        inventory_policy_id=str(
            identity.get("inventory_policy_id", "inventory-policy:vfs-adversarial@1")
        ),
        inventory_policy_revision=str(
            identity.get(
                "inventory_policy_revision", "inventory-policy-revision:1"
            )
        ),
        policy_id=str(identity.get("policy_id", "policy:vfs-symbolic-rollout@1")),
        policy_revision=str(
            identity.get("policy_revision", "sha256:frozen-vfs-symbolic-policy")
        ),
        capability_id=str(
            identity.get("capability_id", "capability:vfs-symbolic-local@1")
        ),
        capability_revision=str(
            identity.get(
                "capability_revision", "sha256:frozen-vfs-symbolic-capability"
            )
        ),
        toolchain_id=str(
            identity.get("toolchain_id", "toolchain:vfs-symbolic-assurance@1")
        ),
        toolchain_revision=str(
            identity.get("toolchain_revision", "toolchain-revision:1")
        ),
        default_mode=str(raw.get("default_mode", "shadow")),
        automatic_mutation_enabled=False,
        authority_flags=authority,
    )

    registry_raw = dict(raw.get("adapter_registry") or {})
    if set(registry_raw) != CLOSED_ADAPTERS:
        raise IpfsKitVfsAssuranceError(
            "adapter_registry must exactly match the closed adapter set"
        )
    adapters: dict[str, AdapterSpec] = {}
    for name, spec in sorted(registry_raw.items()):
        if not isinstance(spec, Mapping):
            raise IpfsKitVfsAssuranceError(f"adapter {name!r} must be an object")
        adapters[name] = AdapterSpec(
            name=name,
            module=str(spec["module"]),
            factory=str(spec["factory"]),
            lazy=bool(spec.get("lazy", True)),
            optional_provider=bool(spec.get("optional_provider", False)),
        )

    roots = tuple(str(item) for item in raw.get("safe_relative_roots") or ())
    if not roots:
        raise IpfsKitVfsAssuranceError("safe_relative_roots must be non-empty")
    if len(roots) != len(set(roots)):
        raise IpfsKitVfsAssuranceError("safe_relative_roots must be unique")
    for root in roots:
        if not _is_safe_relative(root):
            raise IpfsKitVfsAssuranceError(f"unsafe relative root: {root!r}")

    mappings = dict(raw.get("operation_invariant_error_mappings") or {})
    for key in ("operations", "invariants", "error_codes", "canonical_vectors"):
        if key not in mappings:
            raise IpfsKitVfsAssuranceError(
                f"operation_invariant_error_mappings missing {key}"
            )

    cli = dict(raw.get("cli") or {})
    subcommands = tuple(str(item) for item in cli.get("subcommands") or ())
    if set(subcommands) != CLOSED_ADAPTERS:
        raise IpfsKitVfsAssuranceError("cli subcommands must match closed adapters")
    exit_codes = {
        str(k): int(v) for k, v in dict(cli.get("exit_codes") or {}).items()
    }
    for required in ("success", "failure", "usage"):
        if required not in exit_codes:
            raise IpfsKitVfsAssuranceError(f"cli exit_codes missing {required}")

    schemas_extra = {
        k: v
        for k, v in schemas_raw.items()
        if k
        not in {
            "adversarial_e2e_gate",
            "shadow_rollout_report",
            "rollout_decision",
            "control_request",
            "control_result",
            "bounded_status",
            "bounded_findings",
            "bounded_receipts",
            "public_api",
            "version",
        }
    }

    config = IpfsKitVfsAssuranceConfig(
        raw=raw,
        content_id=expected,
        path=str(config_path),
        profile=profile,
        adapters=adapters,
        safe_relative_roots=roots,
        operation_invariant_error_mappings=mappings,
        cli_subcommands=subcommands,
        exit_codes=exit_codes,
        schemas_extra=schemas_extra,
        authority_flags=authority,
    )
    _CONFIG_CACHE[key] = config
    return config


def build_ipfs_kit_vfs_assurance_profile(
    path: Path | str | None = None,
    *,
    checkout_root: Path | None = None,
) -> AssuranceRolloutProfile:
    """Build the locked IPFS Kit VFS assurance profile from config."""

    return load_assurance_config(path, checkout_root=checkout_root).profile


def resolve_safe_root(
    relative: str,
    *,
    checkout_root: Path | None = None,
    config: IpfsKitVfsAssuranceConfig | None = None,
) -> Path:
    """Resolve a relative root under the checkout and the closed allowlist."""

    cfg = config or load_assurance_config(checkout_root=checkout_root)
    if relative not in cfg.safe_relative_roots and relative != ".":
        # Allow any declared root or exact prefix under declared roots.
        allowed = False
        for root in cfg.safe_relative_roots:
            if relative == root or relative.startswith(root.rstrip("/") + "/"):
                allowed = True
                break
        if not allowed:
            raise IpfsKitVfsAssuranceError(f"relative root not allowlisted: {relative!r}")
    if not _is_safe_relative(relative):
        raise IpfsKitVfsAssuranceError(f"unsafe relative root: {relative!r}")
    base = (checkout_root or _repo_root()).resolve()
    target = (base / relative).resolve()
    if base != target and base not in target.parents:
        raise IpfsKitVfsAssuranceError("resolved root escapes checkout")
    return target


def lazy_import_adapter(
    name: str,
    *,
    config: IpfsKitVfsAssuranceConfig | None = None,
) -> Callable[..., Any]:
    """Import a closed adapter factory without loading optional providers."""

    cfg = config or load_assurance_config()
    if name not in cfg.adapters:
        raise IpfsKitVfsAssuranceError(f"unknown adapter: {name}")
    spec = cfg.adapters[name]
    module_root = spec.module.split(".", 1)[0]
    if module_root in _OPTIONAL_PROVIDER_ROOTS or any(
        spec.module.startswith(root + ".") for root in _OPTIONAL_PROVIDER_ROOTS
    ):
        raise IpfsKitVfsAssuranceError(
            f"refusing to import optional provider for adapter {name}"
        )
    module = importlib.import_module(spec.module)
    try:
        factory = getattr(module, spec.factory)
    except AttributeError as exc:
        raise IpfsKitVfsAssuranceError(
            f"adapter factory missing: {spec.module}.{spec.factory}"
        ) from exc
    return factory


def optional_providers_loaded() -> tuple[str, ...]:
    return tuple(sorted(_LOADED_OPTIONAL_PROVIDERS))


def run_rollout(
    *,
    config: IpfsKitVfsAssuranceConfig | None = None,
    desired_mode: str | AssuranceRolloutMode = AssuranceRolloutMode.SHADOW,
    injection: AdversarialInjection | None = None,
) -> dict[str, Any]:
    cfg = config or load_assurance_config()
    return run_symbolic_assurance_e2e(
        profile=cfg.profile,
        desired_mode=desired_mode,
        injection=injection,
    )


def run_verify(
    *,
    config: IpfsKitVfsAssuranceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_assurance_config()
    fixture, report, binding, policy = build_frozen_adversarial_population(
        profile=cfg.profile
    )
    decision = evaluate_symbolic_assurance_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=cfg.profile.default_mode,
    )
    ok = verify_adversarial_e2e_report(report) and verify_symbolic_assurance_rollout(
        decision, report, binding=binding, policy=policy
    )
    return {
        "verified": ok,
        "fixture_cid": fixture.fixture_cid,
        "gate_report_id": report.report_id,
        "decision_id": decision.decision_id,
        "effective_mode": decision.effective_mode.value,
        "automatic_mutation_enabled": False,
        "status": project_bounded_status(decision),
        "findings": project_bounded_findings(decision),
        "receipts": project_bounded_receipts(decision),
    }


def run_inventory(
    *,
    config: IpfsKitVfsAssuranceConfig | None = None,
    checkout_root: Path | None = None,
    relative_root: str = ".",
) -> dict[str, Any]:
    """Lazy inventory pass over an allowlisted relative root."""

    cfg = config or load_assurance_config(checkout_root=checkout_root)
    root = resolve_safe_root(
        relative_root, checkout_root=checkout_root, config=cfg
    )
    factory = lazy_import_adapter("inventory", config=cfg)
    # Build a minimal domain policy for VFS-equivalent inventory when available.
    try:
        from ipfs_accelerate_py.agent_supervisor.analysis.repository_surface_inventory import (
            SurfaceInventoryPolicy,
            SurfaceKindSpec,
            SurfaceSignal,
            SignalTarget,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise IpfsKitVfsAssuranceError("inventory engine unavailable") from exc

    policy = SurfaceInventoryPolicy(
        profile_id="ipfs-kit-vfs-inventory@1",
        schema=str(
            cfg.schemas_extra.get(
                "surface_inventory",
                "ipfs_accelerate_py/agent-supervisor/vfs-surface-inventory@1",
            )
        ),
        contract_version="vfs-surface-inventory/v1",
        content_signals=(
            SurfaceSignal(
                name="domain_content",
                pattern=r"(?i)(?<![a-z0-9])(?:vfs|fsspec|ipfs)(?![a-z0-9])",
                target=SignalTarget.CONTENT,
            ),
        ),
        path_signals=(
            SurfaceSignal(
                name="domain_path",
                pattern=r"(?i)(?:^|[./_-])(?:vfs|fsspec|ipfs)(?:[_-][a-z0-9]+)*(?=[^a-z0-9]|$)",
                target=SignalTarget.PATH,
            ),
        ),
        kind_specs=(
            SurfaceKindSpec(kind="fsspec", combined_patterns=(r"(?i)fsspec",)),
            SurfaceKindSpec(kind="vfs_surface", combined_patterns=(r"(?i)vfs",)),
        ),
    )
    # Scan the resolved root itself (bounded); do not require sibling package names.
    inventory = factory(root, policy, scan_roots=(root,))
    if hasattr(inventory, "to_record"):
        payload = inventory.to_record()
    elif hasattr(inventory, "to_dict"):
        payload = inventory.to_dict()
    else:
        payload = {
            "type": type(inventory).__name__,
            "repr": repr(inventory)[:512],
        }
    return {
        "adapter": "inventory",
        "root": str(root),
        "profile_id": cfg.profile.profile_id,
        "inventory": payload,
        "automatic_mutation_enabled": False,
    }


def run_contracts(
    *,
    config: IpfsKitVfsAssuranceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_assurance_config()
    lazy_import_adapter("contracts", config=cfg)
    mappings = cfg.operation_invariant_error_mappings
    return {
        "adapter": "contracts",
        "profile_id": cfg.profile.profile_id,
        "schemas": {
            "drift_inventory": cfg.schemas_extra.get("drift_inventory"),
            "contract_pack_version": cfg.schemas_extra.get("contract_pack_version"),
        },
        "operations": list(mappings.get("operations", ())),
        "invariants": list(mappings.get("invariants", ())),
        "error_codes": list(mappings.get("error_codes", ())),
        "canonical_vectors": list(mappings.get("canonical_vectors", ())),
        "authority_flags": dict(cfg.authority_flags),
        "automatic_mutation_enabled": False,
    }


def run_differential(
    *,
    config: IpfsKitVfsAssuranceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_assurance_config()
    lazy_import_adapter("differential", config=cfg)
    return {
        "adapter": "differential",
        "profile_id": cfg.profile.profile_id,
        "schema": cfg.schemas_extra.get("differential_witness"),
        "operations": list(
            cfg.operation_invariant_error_mappings.get("operations", ())
        ),
        "automatic_mutation_enabled": False,
    }


def run_parity(
    *,
    config: IpfsKitVfsAssuranceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_assurance_config()
    lazy_import_adapter("parity", config=cfg)
    return {
        "adapter": "parity",
        "profile_id": cfg.profile.profile_id,
        "schema": cfg.schemas_extra.get("parity_report"),
        "automatic_mutation_enabled": False,
    }


def run_benchmark(
    *,
    config: IpfsKitVfsAssuranceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_assurance_config()
    lazy_import_adapter("benchmark", config=cfg)
    return {
        "adapter": "benchmark",
        "profile_id": cfg.profile.profile_id,
        "schema": cfg.schemas_extra.get("benchmark"),
        "automatic_mutation_enabled": False,
    }


def run_pilot(
    *,
    config: IpfsKitVfsAssuranceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_assurance_config()
    lazy_import_adapter("pilot", config=cfg)
    return {
        "adapter": "pilot",
        "profile_id": cfg.profile.profile_id,
        "schema": cfg.schemas_extra.get("pilot"),
        "automatic_mutation_enabled": False,
        "default_mode": (
            cfg.profile.default_mode.value
            if hasattr(cfg.profile.default_mode, "value")
            else str(cfg.profile.default_mode)
        ),
    }


def dispatch(
    command: str,
    *,
    config: IpfsKitVfsAssuranceConfig | None = None,
    checkout_root: Path | None = None,
    relative_root: str = ".",
    desired_mode: str = "shadow",
) -> dict[str, Any]:
    """Dispatch a closed CLI/adapter command."""

    cfg = config or load_assurance_config(checkout_root=checkout_root)
    if command not in CLOSED_ADAPTERS:
        raise IpfsKitVfsAssuranceError(f"unknown command: {command}")
    if command == "inventory":
        return run_inventory(
            config=cfg, checkout_root=checkout_root, relative_root=relative_root
        )
    if command == "contracts":
        return run_contracts(config=cfg)
    if command == "differential":
        return run_differential(config=cfg)
    if command == "parity":
        return run_parity(config=cfg)
    if command == "benchmark":
        return run_benchmark(config=cfg)
    if command == "pilot":
        return run_pilot(config=cfg)
    if command == "rollout":
        return run_rollout(config=cfg, desired_mode=desired_mode)
    if command == "verify":
        return run_verify(config=cfg)
    raise IpfsKitVfsAssuranceError(f"unhandled command: {command}")


def build_public_api(
    *,
    config: IpfsKitVfsAssuranceConfig | None = None,
    injection: AdversarialInjection | None = None,
) -> SymbolicAssurancePublicAPI:
    cfg = config or load_assurance_config()
    _fixture, report, binding, policy = build_frozen_adversarial_population(
        profile=cfg.profile, injection=injection
    )
    return SymbolicAssurancePublicAPI(
        report,
        binding=binding,
        policy=policy,
        initial_mode=cfg.profile.default_mode,
    )


__all__ = (
    "CLOSED_ADAPTERS",
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG_RELATIVE",
    "AdapterSpec",
    "IpfsKitVfsAssuranceConfig",
    "IpfsKitVfsAssuranceError",
    "build_ipfs_kit_vfs_assurance_profile",
    "build_public_api",
    "default_config_path",
    "dispatch",
    "lazy_import_adapter",
    "load_assurance_config",
    "optional_providers_loaded",
    "resolve_safe_root",
    "run_benchmark",
    "run_contracts",
    "run_differential",
    "run_inventory",
    "run_parity",
    "run_pilot",
    "run_rollout",
    "run_verify",
)
