"""Pure, deterministic constraint filtering and catalog candidate ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .schema import (
    CapabilityDescriptor,
    CatalogSnapshot,
    DeploymentDescriptor,
    LifecycleState,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    RouterBinding,
)

MAX_RESOLUTION_CANDIDATES = 1_000
MAX_RESOLUTION_REASONS = 512
_POLICY_KEY = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class ResolutionError(ValueError):
    """A resolution request is malformed."""


def _operation(value: Any) -> Operation:
    if isinstance(value, Operation):
        return value
    try:
        return Operation(value)
    except (TypeError, ValueError) as exc:
        raise ResolutionError("operation is not supported: %r" % (value,)) from exc


def _modality(value: Any) -> Optional[Modality]:
    if value is None:
        return None
    if isinstance(value, Modality):
        return value
    try:
        return Modality(value)
    except (TypeError, ValueError) as exc:
        raise ResolutionError("modality is not supported: %r" % (value,)) from exc


def _selector(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 256:
        raise ResolutionError("%s must be a bounded non-empty string" % field_name)
    return value.strip().casefold()


def _optional_bool(value: Any, field_name: str) -> Optional[bool]:
    if value is not None and not isinstance(value, bool):
        raise ResolutionError("%s must be a boolean or null" % field_name)
    return value


def _policy(value: Any) -> Tuple[Tuple[str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        pairs = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pairs = tuple(value)
    else:
        raise ResolutionError("policy must be an object")
    result = []
    for pair in pairs:
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ResolutionError("policy entries must be key/value pairs")
        key, item = pair
        if not isinstance(key, str):
            raise ResolutionError("policy keys must be strings")
        key = key.strip().casefold()
        if not _POLICY_KEY.fullmatch(key):
            raise ResolutionError("invalid policy key: %s" % key)
        if isinstance(item, bool):
            item = "true" if item else "false"
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            item = str(item)
        if not isinstance(item, str) or len(item.encode("utf-8")) > 256:
            raise ResolutionError("policy values must be bounded scalars")
        result.append((key, item))
    if len(result) > 64 or len({key for key, _ in result}) != len(result):
        raise ResolutionError("policy constraints are duplicated or excessive")
    return tuple(sorted(result))


@dataclass(frozen=True)
class ResolutionRequest:
    """All supported catalog resolution constraints.

    State constraints use exact tri-state semantics: asking for ``True`` does
    not accept an unknown value.  With no state constraint, unknown remains a
    valid candidate and merely ranks below positive observed state.
    """

    operation: Operation
    modality: Optional[Modality] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    deployment: Optional[str] = None
    policy: Tuple[Tuple[str, str], ...] = ()
    device: Optional[str] = None
    context: Optional[int] = None
    health: Optional[bool] = None
    locality: Optional[str] = None
    configured: Optional[bool] = None
    authorized: Optional[bool] = None
    reachable: Optional[bool] = None
    routable: Optional[bool] = None
    limit: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _operation(self.operation))
        object.__setattr__(self, "modality", _modality(self.modality))
        for name in ("model", "provider", "deployment", "device", "locality"):
            object.__setattr__(self, name, _selector(getattr(self, name), name))
        object.__setattr__(self, "policy", _policy(self.policy))
        if self.context is not None and (
            isinstance(self.context, bool)
            or not isinstance(self.context, int)
            or not 1 <= self.context <= 100_000_000
        ):
            raise ResolutionError("context must be between 1 and 100000000")
        for name in ("health", "configured", "authorized", "reachable", "routable"):
            object.__setattr__(self, name, _optional_bool(getattr(self, name), name))
        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or not 1 <= self.limit <= MAX_RESOLUTION_CANDIDATES
        ):
            raise ResolutionError("limit must be between 1 and %d" % MAX_RESOLUTION_CANDIDATES)

    @property
    def context_tokens(self) -> Optional[int]:
        return self.context

    @property
    def require_healthy(self) -> Optional[bool]:
        return self.health

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation.value,
            "modality": None if self.modality is None else self.modality.value,
            "model": self.model,
            "provider": self.provider,
            "deployment": self.deployment,
            "policy": dict(self.policy),
            "device": self.device,
            "context": self.context,
            "health": self.health,
            "locality": self.locality,
            "configured": self.configured,
            "authorized": self.authorized,
            "reachable": self.reachable,
            "routable": self.routable,
            "limit": self.limit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResolutionRequest":
        fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        if not isinstance(data, Mapping) or set(data) - set(fields) or "operation" not in data:
            raise ResolutionError("ResolutionRequest has missing or unknown fields")
        return cls(**dict(data))


# A useful alternate name for API consumers.
ResolutionConstraints = ResolutionRequest


@dataclass(frozen=True)
class ResolutionCandidate:
    provider: ProviderDescriptor
    model: Optional[ModelDescriptor]
    deployment: Optional[DeploymentDescriptor]
    binding: RouterBinding
    score: int
    reasons: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.provider, ProviderDescriptor):
            raise ResolutionError("candidate provider is invalid")
        if self.model is not None and not isinstance(self.model, ModelDescriptor):
            raise ResolutionError("candidate model is invalid")
        if self.deployment is not None and not isinstance(self.deployment, DeploymentDescriptor):
            raise ResolutionError("candidate deployment is invalid")
        if not isinstance(self.binding, RouterBinding):
            raise ResolutionError("candidate binding is invalid")
        if isinstance(self.score, bool) or not isinstance(self.score, int):
            raise ResolutionError("candidate score must be an integer")
        reasons = tuple(self.reasons)
        if len(reasons) > 64 or any(
            not isinstance(item, str) or len(item.encode("utf-8")) > 512 for item in reasons
        ):
            raise ResolutionError("candidate reasons are invalid or excessive")
        object.__setattr__(self, "reasons", reasons)

    @property
    def provider_id(self) -> str:
        return self.provider.provider_id

    @property
    def model_id(self) -> Optional[str]:
        return None if self.model is None else self.model.model_id

    @property
    def deployment_id(self) -> Optional[str]:
        return None if self.deployment is None else self.deployment.deployment_id

    @property
    def binding_id(self) -> str:
        return self.binding.binding_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.to_dict(),
            "model": None if self.model is None else self.model.to_dict(),
            "deployment": (None if self.deployment is None else self.deployment.to_dict()),
            "binding": self.binding.to_dict(),
            "score": self.score,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResolutionCandidate":
        fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        if not isinstance(data, Mapping) or set(data) != set(fields):
            raise ResolutionError("ResolutionCandidate has missing or unknown fields")
        return cls(
            provider=ProviderDescriptor.from_dict(data["provider"]),
            model=(None if data["model"] is None else ModelDescriptor.from_dict(data["model"])),
            deployment=(
                None
                if data["deployment"] is None
                else DeploymentDescriptor.from_dict(data["deployment"])
            ),
            binding=RouterBinding.from_dict(data["binding"]),
            score=data["score"],
            reasons=tuple(data["reasons"]),
        )


@dataclass(frozen=True)
class ResolutionResult:
    request: ResolutionRequest
    candidates: Tuple[ResolutionCandidate, ...]
    reasons: Tuple[str, ...]
    snapshot_revision: str
    total_candidates: int

    def __post_init__(self) -> None:
        if not isinstance(self.request, ResolutionRequest):
            raise ResolutionError("result request is invalid")
        candidates = tuple(self.candidates)
        if len(candidates) > MAX_RESOLUTION_CANDIDATES or any(
            not isinstance(item, ResolutionCandidate) for item in candidates
        ):
            raise ResolutionError("result candidates are invalid or excessive")
        object.__setattr__(self, "candidates", candidates)
        reasons = tuple(self.reasons)
        if len(reasons) > MAX_RESOLUTION_REASONS or any(
            not isinstance(item, str) or len(item.encode("utf-8")) > 512 for item in reasons
        ):
            raise ResolutionError("result reasons are invalid or excessive")
        object.__setattr__(self, "reasons", reasons)
        if (
            isinstance(self.total_candidates, bool)
            or not isinstance(self.total_candidates, int)
            or self.total_candidates < len(candidates)
        ):
            raise ResolutionError("total_candidates is invalid")

    @property
    def found(self) -> bool:
        return bool(self.candidates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "reasons": list(self.reasons),
            "snapshot_revision": self.snapshot_revision,
            "total_candidates": self.total_candidates,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResolutionResult":
        fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        if not isinstance(data, Mapping) or set(data) != set(fields):
            raise ResolutionError("ResolutionResult has missing or unknown fields")
        return cls(
            request=ResolutionRequest.from_dict(data["request"]),
            candidates=tuple(ResolutionCandidate.from_dict(item) for item in data["candidates"]),
            reasons=tuple(data["reasons"]),
            snapshot_revision=data["snapshot_revision"],
            total_candidates=data["total_candidates"],
        )


def _record_matches(record: Any, selector: str) -> bool:
    identity_field = {
        ProviderDescriptor: "provider_id",
        ModelDescriptor: "model_id",
        DeploymentDescriptor: "deployment_id",
        RouterBinding: "binding_id",
    }.get(type(record))
    if identity_field is not None and selector == getattr(record, identity_field):
        return True
    if selector == getattr(record, "name", None):
        return True
    return selector in getattr(record, "aliases", ())


def _select_one(
    records: Iterable[Any], selector: Optional[str], label: str
) -> Tuple[Optional[Any], Optional[str]]:
    if selector is None:
        return None, None
    records = tuple(records)
    identity_field = {
        "provider": "provider_id",
        "model": "model_id",
        "deployment": "deployment_id",
    }[label]
    exact = tuple(record for record in records if getattr(record, identity_field) == selector)
    matches = exact or tuple(record for record in records if _record_matches(record, selector))
    if not matches:
        return None, "%s constraint did not match any canonical record" % label
    ids = {
        getattr(
            item,
            {
                ProviderDescriptor: "provider_id",
                ModelDescriptor: "model_id",
                DeploymentDescriptor: "deployment_id",
            }[type(item)],
        )
        for item in matches
    }
    if len(ids) > 1:
        return None, "%s alias is ambiguous; resolution failed closed" % label
    return matches[0], None


def _candidate_labels(
    provider: ProviderDescriptor,
    model: Optional[ModelDescriptor],
    deployment: Optional[DeploymentDescriptor],
    binding: RouterBinding,
) -> Dict[str, str]:
    result = dict(provider.labels)
    if model is not None:
        result.update(dict(model.labels))
    if deployment is not None:
        result.update(dict(deployment.labels))
    result.update(dict(binding.labels))
    return result


def _candidate_state(
    provider: ProviderDescriptor,
    model: Optional[ModelDescriptor],
    deployment: Optional[DeploymentDescriptor],
    binding: RouterBinding,
) -> OperationalState:
    levels = [binding.state]
    if deployment is not None:
        levels.append(deployment.state)
    if model is not None:
        levels.append(model.state)
    levels.append(provider.state)
    values = {}
    for name in OperationalState.__dataclass_fields__:  # type: ignore[attr-defined]
        values[name] = next(
            (getattr(state, name) for state in levels if getattr(state, name) is not None),
            None,
        )
    return OperationalState(**values)


def _candidate_capabilities(
    provider: ProviderDescriptor,
    model: Optional[ModelDescriptor],
    deployment: Optional[DeploymentDescriptor],
) -> Tuple[CapabilityDescriptor, ...]:
    result = []
    for record in (deployment, model, provider):
        if record is not None:
            result.extend(record.capabilities)
    return tuple(result)


def _context_limit(
    capabilities: Sequence[CapabilityDescriptor], operation: Operation
) -> Optional[int]:
    # Capabilities are ordered most-specific first.  Use the first level that
    # makes a context assertion for the requested operation.
    for capability in capabilities:
        if operation in capability.operations and capability.max_context_tokens is not None:
            return capability.max_context_tokens
    return None


def _modality_matches(
    capabilities: Sequence[CapabilityDescriptor],
    operation: Operation,
    modality: Modality,
) -> bool:
    return any(
        operation in capability.operations
        and (modality in capability.input_modalities or modality in capability.output_modalities)
        for capability in capabilities
    )


def _lifecycle_score(*records: Optional[Any]) -> int:
    weights = {
        LifecycleState.READY: 40,
        LifecycleState.CONFIGURED: 25,
        LifecycleState.DEGRADED: 5,
        LifecycleState.UNAVAILABLE: -40,
        LifecycleState.STOPPED: -50,
        LifecycleState.DEPRECATED: -20,
        LifecycleState.RETIRED: -100,
    }
    return sum(weights.get(item.lifecycle, 0) for item in records if item is not None)


class CatalogResolver:
    """Resolve a catalog snapshot without mutating it or probing providers."""

    def resolve(
        self,
        snapshot: CatalogSnapshot,
        request: Optional[ResolutionRequest] = None,
        **constraints: Any,
    ) -> ResolutionResult:
        if not isinstance(snapshot, CatalogSnapshot):
            raise TypeError("snapshot must be a CatalogSnapshot")
        if request is not None and constraints:
            raise ResolutionError("pass either request or keyword constraints, not both")
        if request is None:
            if "context_tokens" in constraints:
                if "context" in constraints:
                    raise ResolutionError("context and context_tokens are aliases")
                constraints["context"] = constraints.pop("context_tokens")
            if "healthy" in constraints:
                if "health" in constraints:
                    raise ResolutionError("health and healthy are aliases")
                constraints["health"] = constraints.pop("healthy")
            if "require_healthy" in constraints:
                if "health" in constraints:
                    raise ResolutionError("health and require_healthy are aliases")
                constraints["health"] = constraints.pop("require_healthy")
            request = ResolutionRequest(**constraints)
        elif not isinstance(request, ResolutionRequest):
            raise ResolutionError("request must be a ResolutionRequest")

        providers = {item.provider_id: item for item in snapshot.providers}
        models = {item.model_id: item for item in snapshot.models}
        deployments = {item.deployment_id: item for item in snapshot.deployments}
        selected_provider, error = _select_one(snapshot.providers, request.provider, "provider")
        if error:
            return self._empty(snapshot, request, error)

        model_scope = tuple(
            item
            for item in snapshot.models
            if selected_provider is None or item.provider_id == selected_provider.provider_id
        )
        selected_model, error = _select_one(model_scope, request.model, "model")
        if error:
            return self._empty(snapshot, request, error)

        deployment_scope = tuple(
            item
            for item in snapshot.deployments
            if (selected_provider is None or item.provider_id == selected_provider.provider_id)
            and (selected_model is None or item.model_id == selected_model.model_id)
        )
        selected_deployment, error = _select_one(deployment_scope, request.deployment, "deployment")
        if error:
            return self._empty(snapshot, request, error)

        accepted = []
        rejected = []
        for binding in snapshot.bindings:
            provider = providers.get(binding.provider_id)
            deployment = (
                None if binding.deployment_id is None else deployments.get(binding.deployment_id)
            )
            model_id = binding.model_id or (None if deployment is None else deployment.model_id)
            model = None if model_id is None else models.get(model_id)
            reason_prefix = binding.binding_id
            if provider is None:
                rejected.append("%s: missing provider reference" % reason_prefix)
                continue
            if binding.deployment_id is not None and deployment is None:
                rejected.append("%s: missing deployment reference" % reason_prefix)
                continue
            if binding.model_id is not None and model is None:
                rejected.append("%s: missing model reference" % reason_prefix)
                continue
            if deployment is not None and deployment.provider_id != provider.provider_id:
                rejected.append("%s: deployment provider mismatch" % reason_prefix)
                continue
            if (
                deployment is not None
                and binding.model_id is not None
                and deployment.model_id != binding.model_id
            ):
                rejected.append("%s: deployment model mismatch" % reason_prefix)
                continue
            if model is not None and model.provider_id != provider.provider_id:
                rejected.append("%s: model provider mismatch" % reason_prefix)
                continue
            if request.operation not in binding.operations:
                rejected.append("%s: operation mismatch" % reason_prefix)
                continue
            if (
                selected_provider is not None
                and provider.provider_id != selected_provider.provider_id
            ):
                rejected.append("%s: provider mismatch" % reason_prefix)
                continue
            if selected_model is not None and (
                model is None or model.model_id != selected_model.model_id
            ):
                rejected.append("%s: model mismatch" % reason_prefix)
                continue
            if selected_deployment is not None and (
                deployment is None or deployment.deployment_id != selected_deployment.deployment_id
            ):
                rejected.append("%s: deployment mismatch" % reason_prefix)
                continue

            capabilities = _candidate_capabilities(provider, model, deployment)
            if request.modality is not None and not _modality_matches(
                capabilities, request.operation, request.modality
            ):
                rejected.append("%s: modality is unsupported or unknown" % reason_prefix)
                continue
            context_limit = _context_limit(capabilities, request.operation)
            if request.context is not None and (
                context_limit is None or context_limit < request.context
            ):
                rejected.append("%s: context limit is insufficient or unknown" % reason_prefix)
                continue

            labels = _candidate_labels(provider, model, deployment, binding)
            if request.device is not None and labels.get("device", "").casefold() != request.device:
                rejected.append("%s: device mismatch" % reason_prefix)
                continue
            if (
                request.locality is not None
                and labels.get("locality", "").casefold() != request.locality
            ):
                rejected.append("%s: locality mismatch" % reason_prefix)
                continue
            policy_failed = False
            for key, value in request.policy:
                actual = labels.get(key)
                if actual is None and not key.startswith("policy."):
                    actual = labels.get("policy.%s" % key)
                if actual != value:
                    policy_failed = True
                    break
            if policy_failed:
                rejected.append("%s: policy label mismatch" % reason_prefix)
                continue

            state = _candidate_state(provider, model, deployment, binding)
            state_constraints = {
                "healthy": request.health,
                "configured": request.configured,
                "authorized": request.authorized,
                "reachable": request.reachable,
                "routable": request.routable,
            }
            failed_state = next(
                (
                    name
                    for name, wanted in state_constraints.items()
                    if wanted is not None and getattr(state, name) is not wanted
                ),
                None,
            )
            if failed_state is not None:
                rejected.append(
                    "%s: %s state constraint failed or is unknown" % (reason_prefix, failed_state)
                )
                continue

            score = binding.priority * 1_000
            state_weights = {
                "known": 5,
                "configured": 20,
                "authorized": 30,
                "reachable": 50,
                "healthy": 100,
                "routable": 100,
            }
            for name, weight in state_weights.items():
                value = getattr(state, name)
                score += weight if value is True else (-weight if value is False else 0)
            score += _lifecycle_score(provider, model, deployment)
            if labels.get("locality", "").casefold() == "local":
                score += 10
            candidate_reasons = [
                "supports %s through %s" % (request.operation.value, binding.router),
                "binding priority %d" % binding.priority,
            ]
            positive_states = [name for name in state_weights if getattr(state, name) is True]
            if positive_states:
                candidate_reasons.append("positive state: %s" % ", ".join(positive_states))
            if context_limit is not None:
                candidate_reasons.append("context limit %d" % context_limit)
            if request.policy:
                candidate_reasons.append("all policy constraints matched")
            accepted.append(
                ResolutionCandidate(
                    provider=provider,
                    model=model,
                    deployment=deployment,
                    binding=binding,
                    score=score,
                    reasons=tuple(candidate_reasons),
                )
            )

        accepted.sort(
            key=lambda item: (
                -item.score,
                item.provider_id,
                item.model_id or "",
                item.deployment_id or "",
                item.binding_id,
            )
        )
        total = len(accepted)
        candidates = tuple(accepted[: request.limit])
        reasons = tuple(rejected[:MAX_RESOLUTION_REASONS])
        if not candidates:
            summary = "no candidates satisfy the complete constraint intersection"
            reasons = (summary,) + reasons[: MAX_RESOLUTION_REASONS - 1]
        elif total > len(candidates):
            reasons = (
                "candidate result was limited from %d to %d" % (total, len(candidates)),
            ) + reasons[: MAX_RESOLUTION_REASONS - 1]
        return ResolutionResult(
            request=request,
            candidates=candidates,
            reasons=reasons,
            snapshot_revision=snapshot.revision,
            total_candidates=total,
        )

    @staticmethod
    def _empty(
        snapshot: CatalogSnapshot, request: ResolutionRequest, reason: str
    ) -> ResolutionResult:
        return ResolutionResult(
            request=request,
            candidates=(),
            reasons=(
                "no candidates satisfy the complete constraint intersection",
                reason,
            ),
            snapshot_revision=snapshot.revision,
            total_candidates=0,
        )


def resolve(
    snapshot: CatalogSnapshot, request: Optional[ResolutionRequest] = None, **constraints: Any
) -> ResolutionResult:
    """Convenience wrapper around :class:`CatalogResolver`."""

    return CatalogResolver().resolve(snapshot, request, **constraints)


resolve_candidates = resolve
Resolver = CatalogResolver
ResolveRequest = ResolutionRequest
ResolveResult = ResolutionResult
ResolvedCandidate = ResolutionCandidate


__all__ = [
    "CatalogResolver",
    "MAX_RESOLUTION_CANDIDATES",
    "MAX_RESOLUTION_REASONS",
    "ResolutionCandidate",
    "ResolutionConstraints",
    "ResolutionError",
    "ResolutionRequest",
    "ResolutionResult",
    "ResolveRequest",
    "ResolveResult",
    "ResolvedCandidate",
    "Resolver",
    "resolve",
    "resolve_candidates",
]
