"""Paid accelerator capabilities using MCP++ Profile H and x402 v2.

The service in this module is deliberately transport neutral.  It puts the
shared Profile H settlement fence immediately in front of the first durable
job/reservation transition.  Consequently a queue, worker, or capacity manager
cannot observe protected work until payment and the independent authorization
policy have both accepted it.
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import re
import sqlite3
import threading
import time
import uuid
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
    RequestContext,
    SellerResult,
    SellerRuntime,
    ProfileHControlPlane,
    http_response,
    libp2p_response,
)
from mcplusplus_profile_h.canonical import canonical_json, cid_for, commitment
from mcplusplus_profile_h.errors import ProfileHError


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_PROTECTED_KINDS = frozenset({"inference", "job", "reservation"})


class AcceleratorPaymentError(ProfileHError):
    """A stable failure raised before accelerator work is started."""


@dataclass(frozen=True, slots=True)
class ComputeTier:
    """A fixed, pre-priced compute envelope.

    ``units`` is intentionally fixed for the first Profile H release.  This
    avoids an open-ended authorization while retaining explicit timeout and
    partial-result rules.
    """

    amount: str
    units: int = 1
    unit: str = "inference"
    max_duration_seconds: int = 300
    operations: tuple[str, ...] = ("inference/run", "jobs/submit", "reservations/create")
    models: tuple[str, ...] = ("*",)
    hardware: tuple[str, ...] = ("cpu",)
    allow_partial_results: bool = False
    unused_amount_rule: str = "non-refundable"

    def __post_init__(self) -> None:
        if not self.amount.isdigit() or (len(self.amount) > 1 and self.amount.startswith("0")):
            raise ValueError("amount must be a canonical atomic-unit integer")
        if self.units < 1 or self.max_duration_seconds < 1:
            raise ValueError("compute units and duration must be positive")
        if not self.operations or any(not _NAME.fullmatch(value) for value in self.operations):
            raise ValueError("tier operations must be valid names")
        if not self.models or not self.hardware:
            raise ValueError("tier must declare model and hardware scopes")
        if self.unused_amount_rule not in {"non-refundable", "refund-unused", "credit-unused"}:
            raise ValueError("unsupported unused amount rule")


# Compatibility name used by service-specific callers.
AcceleratorComputeTier = ComputeTier
FixedComputeTier = ComputeTier


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
        normalized = {str(name): tier for name, tier in self.tiers.items()}
        if not normalized or any(not _NAME.fullmatch(name) for name in normalized):
            raise ValueError("at least one named compute tier is required")
        object.__setattr__(self, "tiers", normalized)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcceleratorPaymentConfig":
        tiers = {
            name: tier if isinstance(tier, ComputeTier) else ComputeTier(**dict(tier))
            for name, tier in dict(value.get("tiers", {})).items()
        }
        return cls(tiers=tiers, **{key: item for key, item in value.items() if key != "tiers"})


# Shorter alias mirrors the naming used by the kit integration.
AcceleratePaymentConfig = AcceleratorPaymentConfig


def default_compute_tiers(amounts: Mapping[str, str]) -> dict[str, ComputeTier]:
    """Return conservative CPU/GPU fixed-price tiers."""
    required = {"inference", "batch", "gpu-reservation"}
    missing = required.difference(amounts)
    if missing:
        raise ValueError(f"missing prices for: {', '.join(sorted(missing))}")
    return {
        "inference": ComputeTier(amounts["inference"], unit="inference", operations=("inference/run",), hardware=("cpu", "webgpu")),
        "batch": ComputeTier(amounts["batch"], units=100, unit="inference", max_duration_seconds=3600, operations=("jobs/submit",), hardware=("cpu", "cuda"), allow_partial_results=True),
        "gpu-reservation": ComputeTier(amounts["gpu-reservation"], units=60, unit="accelerator-minute", max_duration_seconds=3600, operations=("reservations/create",), hardware=("cuda", "rocm"), unused_amount_rule="credit-unused"),
    }


class CatalogSigner:
    """State-local Ed25519 signer.  Worker handoffs never reference this key."""

    def __init__(self, key: Ed25519PrivateKey) -> None:
        self._key = key

    @classmethod
    def generate(cls) -> "CatalogSigner":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, path: str | Path) -> "CatalogSigner":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            raw = Ed25519PrivateKey.generate().private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(raw)
            except FileExistsError:
                raw = path.read_bytes()
        if len(raw) != 32:
            raise ValueError("invalid persisted Ed25519 catalog key")
        os.chmod(path, 0o600)
        return cls(Ed25519PrivateKey.from_private_bytes(raw))

    def sign(self, document: Mapping[str, Any]) -> dict[str, Any]:
        unsigned = dict(document)
        unsigned.pop("signature", None)
        public = self._key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        unsigned.update({"signatureAlg": "Ed25519", "publicKey": base64.b64encode(public).decode("ascii")})
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


class AcceleratorExecutionStore:
    """Durable execution, entitlement, and metering evidence.

    The unique idempotency key is the domain-level fence complementing the
    payment ledger.  A record in ``starting``/``running`` is never silently
    submitted again after a crash.
    """

    TERMINAL = frozenset({"succeeded", "partial", "failed", "cancelled", "expired", "released"})

    def __init__(self, path: str | Path, artifacts: FileCIDArtifactStore, clock_ms: Callable[[], int]) -> None:
        self.path, self.artifacts, self.clock_ms = Path(path), artifacts, clock_ms
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS executions ("
                "idempotency_key TEXT PRIMARY KEY,request_cid TEXT NOT NULL,operation TEXT NOT NULL,"
                "kind TEXT NOT NULL,resource_id TEXT UNIQUE NOT NULL,tier TEXT NOT NULL,status TEXT NOT NULL,"
                "entitlement_cid TEXT NOT NULL,settlement_cid TEXT NOT NULL,usage_cid TEXT,result_cid TEXT,"
                "created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL,detail TEXT)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30, isolation_level=None)

    def begin(self, *, context: RequestContext, operation: str, kind: str, tier_name: str,
              tier: ComputeTier, settlement_cid: str, subject: str, model: str,
              hardware: str) -> tuple[dict[str, Any], bool]:
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM executions WHERE idempotency_key=?", (context.idempotency_key,)).fetchone()
            if row:
                db.rollback()
                record = self._row(row)
                if record["requestCid"] != context.request_cid or record["operation"] != operation:
                    raise AcceleratorPaymentError("H_REQUEST_MISMATCH", "execution key is bound to different work")
                return record, True
            resource_id = f"{kind}-{uuid.uuid4().hex}"
            entitlement = {
                "schema": "mcp++/profile-h/compute-entitlement@1.0",
                "profileG": "mcp++/risk-scheduling@1.0",
                "createdAt": self.clock_ms(), "parents": [settlement_cid],
                "settlementCid": settlement_cid, "requestCid": context.request_cid,
                "subjectCommitment": commitment(subject), "operation": operation,
                "resourceId": resource_id, "tier": tier_name, "units": tier.units,
                "unit": tier.unit, "model": model, "hardware": hardware,
                "expiresAt": self.clock_ms() + tier.max_duration_seconds * 1000,
                "authority": {"canExecute": True, "canSignPayments": False},
            }
            entitlement_cid = self.artifacts.put(entitlement)
            now = self.clock_ms()
            db.execute(
                "INSERT INTO executions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (context.idempotency_key, context.request_cid, operation, kind, resource_id,
                 tier_name, "starting", entitlement_cid, settlement_cid, None, None, now, now, None),
            )
            db.commit()
        return self.get(resource_id) or {}, False

    @staticmethod
    def _row(row: tuple[Any, ...]) -> dict[str, Any]:
        return dict(zip(("idempotencyKey", "requestCid", "operation", "kind", "resourceId", "tier", "status",
                         "entitlementCid", "settlementCid", "usageRecordCid", "resultCid", "createdAt", "updatedAt", "detail"), row))

    def get(self, resource_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM executions WHERE resource_id=?", (resource_id,)).fetchone()
        return self._row(row) if row else None

    def get_by_key(self, key: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM executions WHERE idempotency_key=?", (key,)).fetchone()
        return self._row(row) if row else None

    def get_by_evidence(self, cid: str) -> dict[str, Any] | None:
        """Resolve only domain-owned public evidence, never arbitrary blocks."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM executions WHERE entitlement_cid=? OR usage_cid=?", (cid, cid)
            ).fetchone()
        return self._row(row) if row else None

    def handoff(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Minimal Profile G worker input, intentionally without payment data."""
        return {"profile": "mcp++/risk-scheduling@1.0", "resourceId": record["resourceId"],
                "operation": record["operation"], "entitlementCid": record["entitlementCid"],
                "requestCid": record["requestCid"], "tier": record["tier"]}

    def mark_running(self, resource_id: str) -> None:
        self._transition(resource_id, "running", allowed={"starting"})

    def finish(self, resource_id: str, outcome: str, result: Any, *, units: int,
               allow_partial: bool) -> dict[str, Any]:
        if outcome not in {"succeeded", "partial", "failed"}:
            raise AcceleratorPaymentError("H_REQUEST_MISMATCH", "invalid execution outcome")
        if outcome == "partial" and not allow_partial:
            outcome = "failed"
        record = self.get(resource_id)
        if not record:
            raise AcceleratorPaymentError("H_RECONCILIATION_REQUIRED", "execution record is missing", retryable=True)
        usage = {
            "schema": "mcp++/profile-h/compute-usage@1.0", "createdAt": self.clock_ms(),
            "parents": [record["entitlementCid"], record["settlementCid"]],
            "entitlementCid": record["entitlementCid"], "requestCid": record["requestCid"],
            "resourceId": resource_id, "outcome": outcome, "units": units,
            "inputCommitment": commitment({"requestCid": record["requestCid"]}),
            "outputCommitment": commitment(result),
        }
        usage_cid, result_cid = self.artifacts.put(usage), commitment(result)
        with self._lock, self._connect() as db:
            changed = db.execute(
                "UPDATE executions SET status=?,usage_cid=?,result_cid=?,updated_at=? "
                "WHERE resource_id=? AND status IN ('starting','running')",
                (outcome, usage_cid, result_cid, self.clock_ms(), resource_id),
            ).rowcount
        if not changed:
            latest = self.get(resource_id)
            if not latest or latest["status"] != outcome:
                raise AcceleratorPaymentError("H_RECONCILIATION_REQUIRED", "execution has a conflicting terminal outcome")
        return self.get(resource_id) or {}

    def cancel(self, resource_id: str, *, reason: str = "caller-requested") -> dict[str, Any]:
        record = self.get(resource_id)
        if not record:
            raise AcceleratorPaymentError("H_EXECUTION_NOT_FOUND", "execution is unknown")
        if record["status"] in self.TERMINAL:
            return record
        cancellation = {
            "schema": "mcp++/profile-h/compute-cancellation@1.0", "createdAt": self.clock_ms(),
            "parents": [record["entitlementCid"]], "resourceId": resource_id,
            "entitlementCid": record["entitlementCid"], "reason": reason[:128],
            "unusedAmountRule": "tier-policy",
        }
        usage_cid = self.artifacts.put(cancellation)
        with self._connect() as db:
            db.execute("UPDATE executions SET status='cancelled',usage_cid=?,detail=?,updated_at=? WHERE resource_id=? AND status NOT IN ('succeeded','partial','failed','cancelled','expired','released')",
                       (usage_cid, reason[:128], self.clock_ms(), resource_id))
        return self.get(resource_id) or {}

    def _transition(self, resource_id: str, status: str, *, allowed: set[str]) -> None:
        placeholders = ",".join("?" for _ in allowed)
        with self._connect() as db:
            changed = db.execute(f"UPDATE executions SET status=?,updated_at=? WHERE resource_id=? AND status IN ({placeholders})",
                                 (status, self.clock_ms(), resource_id, *sorted(allowed))).rowcount
        if not changed:
            current = self.get(resource_id)
            if not current or current["status"] != status:
                raise AcceleratorPaymentError("H_RECONCILIATION_REQUIRED", "invalid execution state transition", retryable=True)

    def reconcile(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM executions WHERE status IN ('starting','running') ORDER BY created_at").fetchall()
        return [{**self._row(row), "reconciliationRequired": True} for row in rows]

    def diagnostics(self) -> dict[str, int]:
        with self._connect() as db:
            return {str(status): int(count) for status, count in db.execute("SELECT status,count(*) FROM executions GROUP BY status")}


class PaidAcceleratorService:
    """Profile H facade for inference, jobs, and resource reservations."""

    ROUTES = {
        ("POST", "/mcp/accelerate/inference"): "inference/run",
        ("POST", "/mcp/accelerate/jobs"): "jobs/submit",
        ("POST", "/mcp/accelerate/reservations"): "reservations/create",
        ("POST", "/mcp/tools/inference/run"): "inference/run",
        ("POST", "/mcp/tools/jobs/submit"): "jobs/submit",
        ("POST", "/mcp/tools/reservations/create"): "reservations/create",
    }
    KINDS = {"inference/run": "inference", "jobs/submit": "job", "reservations/create": "reservation"}

    def __init__(self, config: AcceleratorPaymentConfig, state_dir: str | Path, facilitator: Any, *,
                 signer: CatalogSigner | None = None, clock_ms: Callable[[], int] | None = None,
                 control_mode: str | None = None) -> None:
        self.config, self.state_dir = config, Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.signer = signer or CatalogSigner.load_or_create(self.state_dir / "catalog-signing.key")
        self.artifacts = FileCIDArtifactStore(self.state_dir / "artifacts")
        capabilities = []
        for tier_name, tier in config.tiers.items():
            requirement = PaymentRequirement(config.scheme, config.network, config.asset, tier.amount, config.pay_to,
                                             extra={"tier": tier_name, "units": tier.units, "unit": tier.unit})
            for operation in tier.operations:
                capabilities.append(PaidCapability(f"tool:{operation}:{tier_name}", (requirement,), metadata={
                    "ability": f"tool:{operation}", "operation": operation, "tier": tier_name,
                    "units": tier.units, "unit": tier.unit, "maxDurationSeconds": tier.max_duration_seconds,
                    "models": list(tier.models), "hardware": list(tier.hardware),
                    "allowPartialResults": tier.allow_partial_results,
                    "unusedAmountRule": tier.unused_amount_rule,
                }))
        catalog = CapabilityCatalog(capabilities, version=config.catalog_version)
        self.runtime = SellerRuntime(
            PaymentPolicyEngine(catalog, unlisted=Decision.FREE if config.unlisted_free else Decision.DENIED),
            DuckDBPaymentLedger(self.state_dir / "payments.duckdb"), facilitator, self.artifacts,
            seller_did=config.seller_did, descriptor_cid=config.descriptor_cid, clock_ms=self.clock_ms,
        )
        self.executions = AcceleratorExecutionStore(self.state_dir / "executions.sqlite3", self.artifacts, self.clock_ms)
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
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        document = {"schema": "mcp++/profile-h/accelerator-catalog@1.0", "createdAt": self.clock_ms(),
                    "sellerDid": self.config.seller_did, "descriptorCid": self.config.descriptor_cid,
                    "pricingModel": "fixed-compute-tiers", **self.runtime.policy.catalog.public_document()}
        signed = self.signer.sign(document)
        signed["signedCatalogCid"] = cid_for(signed)
        self.artifacts.put({key: value for key, value in signed.items() if key != "signedCatalogCid"})
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(signed, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(path)
        return signed

    def catalog(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._catalog))

    def _scope(self, operation: str, context: RequestContext, params: Mapping[str, Any]) -> tuple[str, ComputeTier, str, str]:
        tier_name = str(params.get("tier", ""))
        tier = self.config.tiers.get(tier_name)
        if tier is None or operation not in tier.operations:
            raise AcceleratorPaymentError("H_PAYMENT_POLICY_DENIED", "operation is unavailable in the selected tier")
        model, hardware = str(params.get("model", "unspecified")), str(params.get("hardware", tier.hardware[0]))
        if not _NAME.fullmatch(model) or not _NAME.fullmatch(hardware):
            raise AcceleratorPaymentError("H_REQUEST_MISMATCH", "invalid model or hardware identifier")
        allowed_models = tuple(context.attributes.get("models", tier.models))
        allowed_hardware = tuple(context.attributes.get("hardware", tier.hardware))
        if (("*" not in tier.models and model not in tier.models) or ("*" not in allowed_models and model not in allowed_models)
                or hardware not in tier.hardware or hardware not in allowed_hardware):
            raise AcceleratorPaymentError("H_PAYMENT_POLICY_DENIED", "model or hardware policy denied")
        requested_units = params.get("units", tier.units)
        duration = params.get("duration_seconds", tier.max_duration_seconds)
        if isinstance(requested_units, bool) or not isinstance(requested_units, int) or requested_units != tier.units:
            raise AcceleratorPaymentError("H_PAYMENT_POLICY_DENIED", "only the fixed tier compute quantity is accepted")
        if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= tier.max_duration_seconds:
            raise AcceleratorPaymentError("H_PAYMENT_POLICY_DENIED", "requested duration exceeds the fixed tier")
        return tier_name, tier, model, hardware

    def _commercial_binding(self, operation: str, context: RequestContext,
                            params: Mapping[str, Any]) -> CommercialBinding:
        tier_name, _tier, _model, _hardware = self._scope(operation, context, params)
        return CommercialBinding(f"tool:{operation}:{tier_name}", context)

    def _control_evidence(self, kind: str, cid: str, context: RequestContext) -> Mapping[str, Any] | None:
        record = self.executions.get_by_evidence(cid)
        field = "entitlementCid" if kind == "entitlement" else "usageRecordCid"
        if not record or record.get(field) != cid:
            return None
        return self.artifacts.get(cid)

    async def profile_h(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch one complete Profile H control-plane operation."""
        return await self.control_plane.dispatch(method, params)

    async def dispatch(self, operation: str, context: RequestContext, params: Mapping[str, Any],
                       effect: Callable[..., Any | Awaitable[Any]], *, payment: PaymentContext | None = None) -> SellerResult:
        tier_name, tier, model, hardware = self._scope(operation, context, params)
        operation_key = f"tool:{operation}:{tier_name}"

        async def start_after_payment() -> Any:
            ledger_entry = self.runtime.ledger.get(context.idempotency_key)
            if ledger_entry is None or ledger_entry.state not in {"settled", "executing"} or not ledger_entry.settlement_cid:
                raise AcceleratorPaymentError("H_RECONCILIATION_REQUIRED", "payment settlement fence is absent", retryable=True)
            record, existed = self.executions.begin(
                context=context, operation=operation, kind=self.KINDS[operation], tier_name=tier_name,
                tier=tier, settlement_cid=ledger_entry.settlement_cid,
                subject=str(context.attributes.get("subject", "anonymous")), model=model, hardware=hardware,
            )
            if existed:
                if record["status"] in AcceleratorExecutionStore.TERMINAL:
                    return {"resourceId": record["resourceId"], "jobId": record["resourceId"] if record["kind"] == "job" else None,
                            "reservationId": record["resourceId"] if record["kind"] == "reservation" else None,
                            "entitlementCid": record["entitlementCid"], "usageRecordCid": record["usageRecordCid"],
                            "outcome": record["status"], "recovered": True}
                raise AcceleratorPaymentError("H_RECONCILIATION_REQUIRED", "work may already have started", retryable=True)
            self.executions.mark_running(record["resourceId"])
            handoff = self.executions.handoff(record)
            try:
                value = await self._invoke_effect(effect, handoff)
            except Exception as exc:
                self.executions.finish(record["resourceId"], "failed", {"errorType": type(exc).__name__}, units=tier.units,
                                       allow_partial=tier.allow_partial_results)
                raise
            outcome = "succeeded"
            if isinstance(value, Mapping) and value.get("outcome") in {"succeeded", "partial", "failed"}:
                outcome = str(value["outcome"])
            completed = self.executions.finish(record["resourceId"], outcome, value, units=tier.units,
                                               allow_partial=tier.allow_partial_results)
            envelope = {"resourceId": record["resourceId"], "entitlementCid": record["entitlementCid"],
                        "usageRecordCid": completed["usageRecordCid"], "outcome": completed["status"]}
            if record["kind"] == "job": envelope["jobId"] = record["resourceId"]
            if record["kind"] == "reservation": envelope["reservationId"] = record["resourceId"]
            return {**dict(value), **envelope} if isinstance(value, Mapping) else {"result": value, **envelope}

        return await self.runtime.dispatch(operation_key, context, start_after_payment, payment=payment)

    @staticmethod
    async def _invoke_effect(effect: Callable[..., Any], handoff: Mapping[str, Any]) -> Any:
        try:
            signature = inspect.signature(effect)
            accepts_argument = any(p.kind in (p.VAR_POSITIONAL, p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                                   for p in signature.parameters.values())
        except (TypeError, ValueError):
            accepts_argument = False
        value = effect(dict(handoff)) if accepts_argument else effect()
        return await value if inspect.isawaitable(value) else value

    def cancel(self, resource_id: str, context: RequestContext, *, reason: str = "caller-requested") -> dict[str, Any]:
        if not context.authorized or not context.policy_allowed:
            raise AcceleratorPaymentError("H_PAYMENT_POLICY_DENIED", "cancellation policy denied")
        record = self.executions.get(resource_id)
        if not record or record["requestCid"] != context.request_cid:
            raise AcceleratorPaymentError("H_EXECUTION_NOT_FOUND", "execution is unknown to this request")
        return self.executions.cancel(resource_id, reason=reason)

    def worker_handoff(self, resource_id: str, context: RequestContext) -> dict[str, Any]:
        if not context.authorized or not context.policy_allowed:
            raise AcceleratorPaymentError("H_PAYMENT_POLICY_DENIED", "worker handoff denied")
        record = self.executions.get(resource_id)
        if not record:
            raise AcceleratorPaymentError("H_EXECUTION_NOT_FOUND", "execution is unknown")
        return self.executions.handoff(record)

    async def handle_http(self, method: str, path: str, context: RequestContext, params: Mapping[str, Any],
                          effect: Callable[..., Any] | None = None, *, payment_header: str | None = None) -> tuple[int, dict[str, str], Any]:
        method = method.upper()
        if method == "GET" and path == "/mcp/payments/catalog":
            return 200, {"ETag": self._catalog["signedCatalogCid"]}, self.catalog()
        for prefix, evidence_field in (("/mcp/payments/entitlements/", "entitlementCid"),
                                       ("/mcp/payments/usage/", "usageRecordCid")):
            if method == "GET" and path.startswith(prefix):
                if not context.authorized or not context.policy_allowed:
                    return 403, {}, {"error": "H_PAYMENT_POLICY_DENIED"}
                evidence_cid = path.removeprefix(prefix)
                record = self.executions.get_by_evidence(evidence_cid)
                if not record or record[evidence_field] != evidence_cid:
                    return 404, {}, {"error": "H_EVIDENCE_NOT_FOUND"}
                artifact = self.artifacts.get(evidence_cid)
                return (200, {}, artifact) if artifact else (404, {}, {"error": "H_EVIDENCE_NOT_FOUND"})
        for prefix, field in (("/mcp/accelerate/jobs/", "job"), ("/mcp/accelerate/reservations/", "reservation")):
            if path.startswith(prefix):
                resource_id = path.removeprefix(prefix).removesuffix("/cancel")
                if not context.authorized or not context.policy_allowed:
                    return 403, {}, {"error": "H_PAYMENT_POLICY_DENIED"}
                if method == "POST" and path.endswith("/cancel"):
                    try:
                        return 200, {}, self.cancel(resource_id, context, reason=str(params.get("reason", "caller-requested")))
                    except AcceleratorPaymentError as error:
                        return 404 if error.code == "H_EXECUTION_NOT_FOUND" else 403, {}, {"error": error.code}
                record = self.executions.get(resource_id)
                if method == "GET" and record and record["kind"] == field:
                    return 200, {}, record
                return 404, {}, {"error": "H_EXECUTION_NOT_FOUND"}
        operation = self.ROUTES.get((method, path))
        if operation is None:
            return 404, {}, {"error": "H_PAYMENT_POLICY_DENIED"}
        if effect is None:
            raise ValueError("a protected accelerator operation requires an effect callback")
        payment = self._decode_payment(payment_header) if payment_header else None
        return http_response(await self.dispatch(operation, context, params, effect, payment=payment))

    async def handle_libp2p(self, request: Mapping[str, Any], context: RequestContext,
                            effect: Callable[..., Any] | None = None) -> dict[str, Any]:
        operation = str(request.get("operation", ""))
        if operation == "mcp++/payments/catalog":
            return {"result": self.catalog()}
        params = request.get("params", {})
        if operation not in self.KINDS or not isinstance(params, Mapping) or effect is None:
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
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise AcceleratorPaymentError("H_INVALID_PAYMENT_MESSAGE", "invalid PAYMENT-SIGNATURE header") from exc

    async def reconcile(self) -> dict[str, Any]:
        return {"payments": await self.runtime.reconcile(), "executions": self.executions.reconcile()}

    async def diagnostics(self) -> dict[str, Any]:
        base = await self.runtime.diagnostics()
        return {**base, "signedCatalogCid": self._catalog["signedCatalogCid"],
                "catalogSignatureValid": CatalogSigner.verify(self._catalog),
                "fixedComputeTiers": sorted(self.config.tiers), "executions": self.executions.diagnostics()}


# Natural spelling and legacy-facing aliases.
PaidAccelerateService = PaidAcceleratorService
AcceleratorService = PaidAcceleratorService
AcceleratePaymentError = AcceleratorPaymentError

__all__ = [
    "AcceleratePaymentConfig", "AcceleratePaymentError", "AcceleratorComputeTier",
    "AcceleratorExecutionStore", "AcceleratorPaymentConfig", "AcceleratorPaymentError",
    "AcceleratorService", "CatalogSigner", "ComputeTier", "FixedComputeTier",
    "PaidAccelerateService", "PaidAcceleratorService",
    "default_compute_tiers",
]

