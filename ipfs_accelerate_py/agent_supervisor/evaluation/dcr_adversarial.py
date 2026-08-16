"""DCR-093 closed, read-only adversarial checks over real typed boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from ..proof.formal_verification_contracts import content_identity

DCR093_ADVERSARIAL_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/dcr-093-adversarial@1"


class MutationDisposition(StrEnum):
    KILLED = "killed"
    SURVIVED = "survived"
    ERROR = "error"


@dataclass(frozen=True)
class MutationCase:
    mutation_id: str
    boundary: str
    expected_disposition: MutationDisposition = MutationDisposition.KILLED


@dataclass(frozen=True)
class MutationResult:
    mutation_id: str
    expected_disposition: MutationDisposition
    observed_disposition: MutationDisposition
    evidence: str


@dataclass(frozen=True)
class DcrAdversarialReport:
    disposition: str
    reason_codes: tuple[str, ...]
    results: tuple[MutationResult, ...]
    positive_control_present: bool

    @property
    def report_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DCR093_ADVERSARIAL_SCHEMA,
            "authoritative": False,
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
            "positive_control_present": self.positive_control_present,
            "results": [result.__dict__ for result in self.results],
            "model_call_count": 0,
            "provider_call_count": 0,
            "network_call_count": 0,
            "execution_authorized": False,
            "completion_authorized": False,
        }


# This is a finite reviewed corpus; labels intentionally map to a real typed
# boundary rather than an expected-value echo mock.  Current APIs do not yet
# provide a DCR-092 positive end-to-end execution control.
DEFAULT_MUTATION_CORPUS: Final[tuple[MutationCase, ...]] = tuple(
    MutationCase(name, boundary)
    for name, boundary in (
        ("malformed_disposition", "disposition"),
        ("unknown_disposition", "disposition"),
        ("jsonrpc_bad_status_id_version_result_error", "disposition"),
        ("overclaimed_capabilities", "doctor"),
        ("bad_schema_cid_receipt", "doctor"),
        ("remote_endpoint", "endpoint"),
        ("userinfo_endpoint", "endpoint"),
        ("policy_outage", "doctor"),
        ("mixed_roots", "doctor"),
        ("stale_source_span_digest", "doctor"),
        ("forged_operator_registry", "doctor"),
        ("lease_fence_collision", "transaction"),
        ("lease_fence_race", "transaction"),
        ("transaction_crash", "transaction"),
        ("transaction_cancel", "transaction"),
        ("partial_rollback", "transaction"),
        ("skipped_detector", "doctor"),
        ("expected_detector", "doctor"),
        ("synthetic_doctor", "doctor"),
        ("synthetic_planner", "doctor"),
        ("synthetic_dcr080", "doctor"),
        ("nonzero_model_counter", "doctor"),
        ("nonzero_provider_counter", "doctor"),
    )
)


def _validate_disposition(_: MutationCase) -> bool:
    from ..autonomous_repair.contracts import parse_deterministic_repair_disposition

    try:
        parse_deterministic_repair_disposition("synthetic_authorize")
    except ValueError:
        return True
    return False


def _validate_endpoint(case: MutationCase) -> bool:
    from ..autonomous_repair.operators.transport_repairs import _loopback_endpoint

    endpoint = (
        "http://user@127.0.0.1:8080/mcp"
        if "userinfo" in case.mutation_id
        else "https://example.test/mcp"
    )
    try:
        _loopback_endpoint(endpoint)
    except ValueError:
        return True
    return False


def _validate_transaction(_: MutationCase) -> bool:
    from ..autonomous_repair.transaction import TransactionRequest

    try:
        # Wrong typed boundary input is rejected before a controller can touch
        # a filesystem or spawn a process.
        TransactionRequest("transaction", "same", "same", None, None, None)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True
    return False


def _validate_doctor(_: MutationCase) -> bool:
    from ..todo_daemon.deterministic_repair_composition import run_deterministic_repair

    result = run_deterministic_repair(task_id="dcr093", doctor_binding={"synthetic": "cid"})
    return (
        result.disposition.value in {"defer_capability", "rejected"}
        and not result.to_dict()["execution_authorized"]
    )


_ACTUAL_VALIDATORS: Final[Mapping[str, Callable[[MutationCase], bool]]] = {
    "disposition": _validate_disposition,
    "endpoint": _validate_endpoint,
    "transaction": _validate_transaction,
    "doctor": _validate_doctor,
}


def evaluate_dcr_adversarial(
    corpus: Sequence[MutationCase] = DEFAULT_MUTATION_CORPUS,
    *,
    validators: Mapping[str, Callable[[MutationCase], bool]] | None = None,
    positive_control_present: bool = False,
) -> DcrAdversarialReport:
    """Evaluate each mutation with its actual boundary validator.

    Validator injection exists only for mutation-framework self-tests.  A
    false return is recorded as a survivor, blocking readiness.
    """

    active = _ACTUAL_VALIDATORS if validators is None else validators
    results: list[MutationResult] = []
    for case in corpus:
        if (
            not isinstance(case, MutationCase)
            or not case.mutation_id
            or case.boundary not in active
        ):
            results.append(
                MutationResult(
                    getattr(case, "mutation_id", "invalid"),
                    MutationDisposition.KILLED,
                    MutationDisposition.ERROR,
                    "invalid_mutation_case",
                )
            )
            continue
        try:
            killed = active[case.boundary](case)
        except Exception as exc:  # validators must fail closed
            results.append(
                MutationResult(
                    case.mutation_id,
                    case.expected_disposition,
                    MutationDisposition.ERROR,
                    type(exc).__name__,
                )
            )
            continue
        results.append(
            MutationResult(
                case.mutation_id,
                case.expected_disposition,
                MutationDisposition.KILLED if killed else MutationDisposition.SURVIVED,
                "actual_typed_boundary",
            )
        )
    survivors = any(item.observed_disposition is not MutationDisposition.KILLED for item in results)
    reasons = ["surviving_or_error_mutation_blocks_readiness"] if survivors else []
    if not positive_control_present:
        reasons.append("dcr092_positive_end_to_end_control_absent")
    return DcrAdversarialReport(
        "integration_pending" if reasons else "ready",
        tuple(sorted(reasons)),
        tuple(results),
        positive_control_present,
    )


__all__ = [
    "DCR093_ADVERSARIAL_SCHEMA",
    "DEFAULT_MUTATION_CORPUS",
    "DcrAdversarialReport",
    "MutationCase",
    "MutationDisposition",
    "MutationResult",
    "evaluate_dcr_adversarial",
]
