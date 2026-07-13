"""Paid accelerator capabilities using MCP++ Profile H and x402 v2.

This module keeps the commercial boundary immediately in front of the first
compute admission.  The callback passed to :meth:`dispatch` is not invoked
until the shared Profile H runtime has durably settled the request.
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from mcplusplus_profile_h import (
    CallbackFacilitator,
    CapabilityCatalog,
    CommercialBinding,
    Decision,
    DuckDBPaymentLedger,
    FileCIDArtifactStore,
    PaidCapability,
    PaymentContext,
    PaymentPolicyEngine,
    PaymentRequirement,
    ProfileHControlPlane,
    RequestContext,
    SellerResult,
    SellerRuntime,
    http_response,
    libp2p_response,
)
from mcplusplus_profile_h.canonical import canonical_json, cid_for, commitment
from mcplusplus_profile_h.errors import ProfileHError


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_OPERATIONS = frozenset({"inference/run", "jobs/submit", "reservations/create"})


class AcceleratorPaymentError(ProfileHError):
    """Stable failure raised before a worker can observe protected work."""


@dataclass(frozen=True, slots=True)
class ComputeTier:
    """A fixed, bounded compute offer.

    Profile H starts with fixed-price, fixed-unit admissions. Metered work uses
    the shared ``upto`` implementation and must not be silently substituted
    for this offer.
    """

    amount: str
    units: int = 1
    unit: str = "inference"
    max_duration_seconds: int = 300
    operations: tuple[str, ...] = ("inference/run",)
    models: tuple[str, ...] = ()
    hardware: tuple[str, ...] = ()
    allow_partial_results: bool = False
    unused_amount_rule: str = "non-refundable"

    def __post_init__(self) -> None:
        if not self.amount.isdigit() or (len(self.amount) > 1 and self.amount.startswith("0")):
            raise ValueError("amount must be a canonical atomic-unit integer")
        if self.units < 1 or self.max_duration_seconds < 1 or not self.unit:
            raise ValueError("compute units, duration, and unit must be positive")
        operations = tuple(dict.fromkeys(str(item) for item in self.operations))
        if not operations or not set(operations).issubset(_OPERATIONS):
            raise ValueError("tier operations must be valid names")
        models = tuple(dict.fromkeys(str(item) for item in self.models))
        hardware = tuple(dict.fromkeys(str(item) for item in self.hardware))
        if not models or not hardware or any(not _IDENTIFIER.fullmatch(item) for item in (*models, *hardware)):
            raise ValueError("tier must declare valid model and hardware scopes")
        if self.unused_amount_rule not in {"non-refundable", "refund-unused", "credit-unused"}:
            raise ValueError("unsupported unused amount rule")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "hardware", hardware)


@dataclass(frozen=True, slots=True)
class AcceleratorPaymentConfig:
    seller_did: str
    descriptor_cid: str
    pay_to: str
    asset: str
    tiers: Mapping[str, ComputeTier]
    network: str = "eip155:84532"
    scheme: str = "exact"
    catalog_version: str = "1"
    unlisted_free: bool = True

    def __post_init__(self) -> None:
        if not self.seller_did.startswith("did:") or not self.descriptor_cid:
            raise ValueError("seller_did and descriptor_cid are required")
        if ":" not in self.network or not self.pay_to or not self.asset:
            raise ValueError("valid network, asset, and payee are required")
        tiers = {str(name): item for name, item in self.tiers.items()}
        if not tiers or any(not _IDENTIFIER.fullmatch(name) for name in tiers):
            raise ValueError("at least one named compute tier is required")
        if any(not isinstance(value, ComputeTier) for value in tiers.values()):
            raise TypeError("tiers must contain ComputeTier values")
        object.__setattr__(self, "tiers", tiers)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcceleratorPaymentConfig":
        tiers = {
            name: item if isinstance(item, ComputeTier) else ComputeTier(**dict(item))
            for name, item in dict(value.get("tiers", {})).items()
        }
        return cls(tiers=tiers, **{key: item for key, item in value.items() if key != "tiers"})


class CatalogSigner:
    """State-local Ed25519 catalog signer; payment credentials are never stored."""

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._key = private_key

    @classmethod
    def generate(cls) -> "CatalogSigner":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, path: str | Path) -> "CatalogSigner":
        key_path = Path(path)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = key_path.read_bytes()
        except FileNotFoundError:
            raw = Ed25519PrivateKey.generate().private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
            try:
                descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(raw)
            except FileExistsError:
                raw = key_path.read_bytes()
        if len(raw) != 32:
            raise ValueError("invalid persisted Ed25519 catalog key")
        os.chmod(key_path, 0o600)
        return cls(Ed25519PrivateKey.from_private_bytes(raw))

    def sign(self, document: Mapping[str, Any]) -> dict[str, Any]:
        unsigned = dict(document)
        unsigned.pop("signature", None)
        unsigned.update({
            "signatureAlg": "Ed25519",
            "publicKey": base64.b64encode(self._key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode("ascii"),
        })
        return {**unsigned, "signature": base64.b64encode(self._key.sign(canonical_json(unsigned))).decode("ascii")}

    @staticmethod
    def verify(document: Mapping[str, Any]) -> bool:
        try:
            unsigned = dict(document)
            unsigned.pop("signedCatalogCid", None)
            signature = base64.b64decode(unsigned.pop("signature"), validate=True)
            public = base64.b64decode(unsigned["publicKey"], validate=True)
            Ed25519PublicKey.from_public_bytes(public).verify(signature, canonical_json(unsigned))
            return True
        except (InvalidSignature, KeyError, TypeError, ValueError):
            return False


class PaidAcceleratorService:
    """Transport-neutral paid facade for inference, jobs, and reservations."""

    ROUTES = {
        ("POST", "/mcp/accelerate/inference"): "inference/run",
        ("POST", "/mcp/accelerate/jobs"): "jobs/submit",
        ("POST", "/mcp/accelerate/reservations"): "reservations/create",
    }

    def __init__(
        self,
        config: AcceleratorPaymentConfig,
        state_dir: str | Path,
        facilitator: Any,
        *,
        signer: CatalogSigner | None = None,
        clock_ms: Callable[[], int] | None = None,
        control_mode: str | None = None,
    ) -> None:
        self.config = config
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.signer = signer or CatalogSigner.load_or_create(self.state_dir / "catalog-signing.key")
        self.artifacts = FileCIDArtifactStore(self.state_dir / "artifacts")
        capabilities: list[PaidCapability] = []
        for tier_name, tier in config.tiers.items():
            requirement = PaymentRequirement(
                config.scheme, config.network, config.asset, tier.amount, config.pay_to,
                extra={"tier": tier_name, "unit": tier.unit, "units": tier.units,
                       "maxDurationSeconds": tier.max_duration_seconds},
            )
            for operation in tier.operations:
                capabilities.append(PaidCapability(
                    f"tool:{operation}:{tier_name}", (requirement,), metadata={
                        "ability": f"tool:{operation}", "tier": tier_name, "unit": tier.unit,
                        "units": tier.units, "models": list(tier.models), "hardware": list(tier.hardware),
                        "maxDurationSeconds": tier.max_duration_seconds,
                        "allowPartialResults": tier.allow_partial_results,
                        "unusedAmountRule": tier.unused_amount_rule,
                    },
                ))
        catalog = CapabilityCatalog(capabilities, version=config.catalog_version)
        self.runtime = SellerRuntime(
            PaymentPolicyEngine(catalog, unlisted=Decision.FREE if config.unlisted_free else Decision.DENIED),
            DuckDBPaymentLedger(self.state_dir / "payments.duckdb"), facilitator, self.artifacts,
            seller_did=config.seller_did, descriptor_cid=config.descriptor_cid, clock_ms=self.clock_ms,
        )
        self._catalog = self._build_catalog()
        mode = control_mode or ("local-test" if isinstance(facilitator, CallbackFacilitator) else "facilitator")
        self.control_plane = ProfileHControlPlane(
            runtime=self.runtime, catalog=self.catalog, bind=self._commercial_binding,
            reconcile=self.reconcile, evidence=self._control_evidence, mode=mode,
            upstream_x402_http_conformance=mode != "local-test",
        )

    def _build_catalog(self) -> dict[str, Any]:
        path = self.state_dir / "signed-accelerator-catalog.json"
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (isinstance(saved, dict) and CatalogSigner.verify(saved)
                    and saved.get("catalogCid") == self.runtime.policy.catalog.cid
                    and saved.get("sellerDid") == self.config.seller_did
                    and saved.get("descriptorCid") == self.config.descriptor_cid):
                return saved
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        document = {
            "schema": "mcp++/profile-h/accelerator-catalog@1.0", "createdAt": self.clock_ms(),
            "sellerDid": self.config.seller_did, "descriptorCid": self.config.descriptor_cid,
            "pricingModel": "fixed-compute-tier", **self.runtime.policy.catalog.public_document(),
        }
        signed = self.signer.sign(document)
        signed["signedCatalogCid"] = cid_for(signed)
        self.artifacts.put({key: value for key, value in signed.items() if key != "signedCatalogCid"})
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(canonical_json(signed))
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        return signed

    def catalog(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._catalog))

    def _scope(self, operation: str, context: RequestContext, params: Mapping[str, Any]) -> tuple[str, ComputeTier]:
        tier_name = str(params.get("tier", ""))
        tier = self.config.tiers.get(tier_name)
        if tier is None or operation not in tier.operations:
            raise AcceleratorPaymentError("H_PAYMENT_POLICY_DENIED", "compute tier does not allow this operation")
        model = str(params.get("model", ""))
        hardware = str(params.get("hardware", ""))
        allowed_models = {str(item) for item in context.attributes.get("models", tier.models)}
        allowed_hardware = {str(item) for item in context.attributes.get("hardware", tier.hardware)}
        if model not in tier.models or model not in allowed_models:
            raise AcceleratorPaymentError("H_PAYMENT_POLICY_DENIED", "model scope denied")
        if hardware not in tier.hardware or hardware not in allowed_hardware:
            raise AcceleratorPaymentError("H_PAYMENT_POLICY_DENIED", "hardware scope denied")
        units = params.get("units", tier.units)
        if isinstance(units, bool) or not isinstance(units, int) or units != tier.units:
            raise AcceleratorPaymentError("H_ENTITLEMENT_EXHAUSTED", "compute units must match the fixed tier")
        duration = params.get("durationSeconds", params.get("duration_seconds", tier.max_duration_seconds))
        if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= tier.max_duration_seconds:
            raise AcceleratorPaymentError("H_ENTITLEMENT_EXHAUSTED", "compute duration exceeds tier bound")
        return tier_name, tier

    def _commercial_binding(self, operation: str, context: RequestContext, params: Mapping[str, Any]) -> CommercialBinding:
        tier_name, _tier = self._scope(operation, context, params)
        clean = RequestContext(context.request_cid, context.idempotency_key, context.authorized,
                               context.policy_allowed, None, context.attributes)
        return CommercialBinding(f"tool:{operation}:{tier_name}", clean)

    def _control_evidence(self, _kind: str, cid: str, context: RequestContext) -> Mapping[str, Any] | None:
        entry = self.runtime.ledger.get_by_artifact(cid)
        if entry is None or entry.request_cid != context.request_cid:
            return None
        return self.artifacts.get(cid)

    async def profile_h(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return await self.control_plane.dispatch(method, params)

    async def dispatch(
        self,
        operation: str,
        context: RequestContext,
        params: Mapping[str, Any],
        effect: Callable[..., Any],
        *,
        payment: PaymentContext | None = None,
    ) -> SellerResult:
        binding = self._commercial_binding(operation, context, params)

        async def guarded_effect() -> Any:
            try:
                accepts_argument = any(
                    item.kind in (item.VAR_POSITIONAL, item.POSITIONAL_ONLY, item.POSITIONAL_OR_KEYWORD)
                    for item in inspect.signature(effect).parameters.values()
                )
            except (TypeError, ValueError):
                accepts_argument = False
            handoff = {"operation": operation, "tier": str(params.get("tier")), "requestCid": context.request_cid}
            value = effect(handoff) if accepts_argument else effect()
            return await value if hasattr(value, "__await__") else value

        return await self.runtime.dispatch(binding.operation, binding.context, guarded_effect, payment=payment)

    async def handle_http(
        self,
        method: str,
        path: str,
        context: RequestContext,
        params: Mapping[str, Any],
        effect: Callable[..., Any] | None = None,
        *,
        payment_header: str | None = None,
    ) -> tuple[int, dict[str, str], Any]:
        method = method.upper()
        if method == "GET" and path == "/mcp/payments/catalog":
            return 200, {"ETag": self._catalog["signedCatalogCid"]}, self.catalog()
        operation = self.ROUTES.get((method, path))
        if operation is None:
            return 404, {}, {"error": "H_PAYMENT_POLICY_DENIED"}
        if effect is None:
            raise ValueError("a protected accelerator operation requires an effect callback")
        payment = self._decode_payment(payment_header) if payment_header else None
        return http_response(await self.dispatch(operation, context, params, effect, payment=payment))

    async def handle_libp2p(
        self,
        request: Mapping[str, Any],
        context: RequestContext,
        effect: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        operation = str(request.get("operation", ""))
        if operation.startswith("mcp++/payments/"):
            return {"result": await self.profile_h(operation, request.get("params", {}))}
        if operation == "mcp++/payments/catalog":
            return {"result": self.catalog()}
        params = request.get("params", {})
        if not isinstance(params, Mapping) or effect is None:
            return {"error": {"code": "H_PAYMENT_POLICY_DENIED"}}
        raw = request.get("payment_context")
        payment = None
        if isinstance(raw, Mapping):
            payment = PaymentContext(raw.get("payload", {}), str(raw.get("quoteCid", "")),
                                     str(raw.get("requestCid", "")), int(raw.get("requirementIndex", 0)))
        return libp2p_response(await self.dispatch(operation, context, params, effect, payment=payment))

    @staticmethod
    def _decode_payment(value: str) -> PaymentContext:
        try:
            data = json.loads(base64.b64decode(value, validate=True))
            return PaymentContext(data["payload"], data["quoteCid"], data["requestCid"], int(data.get("requirementIndex", 0)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise AcceleratorPaymentError("H_INVALID_PAYMENT_MESSAGE", "invalid PAYMENT-SIGNATURE header") from error

    async def reconcile(self) -> list[dict[str, Any]]:
        return await self.runtime.reconcile()

    async def diagnostics(self) -> dict[str, Any]:
        base = await self.runtime.diagnostics()
        return {**base, "signedCatalogCid": self._catalog["signedCatalogCid"],
                "catalogSignatureValid": CatalogSigner.verify(self._catalog),
                "computeTiers": sorted(self.config.tiers)}


PaidAccelerateService = PaidAcceleratorService
AcceleratorService = PaidAcceleratorService
AcceleratePaymentError = AcceleratorPaymentError

__all__ = [
    "AcceleratePaymentError", "AcceleratorPaymentConfig", "AcceleratorPaymentError",
    "AcceleratorService", "CatalogSigner", "ComputeTier", "PaidAccelerateService",
    "PaidAcceleratorService",
]
