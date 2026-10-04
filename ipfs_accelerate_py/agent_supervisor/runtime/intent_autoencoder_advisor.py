"""Optional datasets-owned intent inference before goal decomposition.

The original instruction and independently admitted domain constraints remain
authoritative inputs. This adapter can only supply unverified planning advice;
an unavailable, damaged, or untrained optional model never blocks planning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

LEGACY_SCHEMA = "ipfs-accelerate-intent-preplanning-advice@1"
SCHEMA = "ipfs-accelerate-intent-preplanning-advice@2"
MAX_DESCRIPTOR_BYTES = 32_768
MAX_REPORT_BYTES = 262_144
AUTHORITY_FIELDS = ("proof_authority", "execution_authority", "completion_authority", "omission_authority")
ROUNDTRIP_CHECKPOINT_SCHEMA = "intent-roundtrip-checkpoint/v1"
COPY_CHECKPOINT_SCHEMA = "intent-copy-roundtrip-checkpoint/v1"
ROUNDTRIP_REPORT_SCHEMA = "intent-instruction-roundtrip/v1"
COPY_REPORT_SCHEMA = "intent-instruction-copy-roundtrip/v1"
EXTENDED_REPORT_SCHEMA = "intent-instruction-extended-roundtrip/v1"
CONTEXT_REPORT_SCHEMA = "intent-instruction-extended-roundtrip/v2"
EXTENDED_REPORT_SCHEMAS = frozenset({EXTENDED_REPORT_SCHEMA, CONTEXT_REPORT_SCHEMA})
SEMANTIC_REPORT_SCHEMAS = frozenset({ROUNDTRIP_REPORT_SCHEMA, COPY_REPORT_SCHEMA, *EXTENDED_REPORT_SCHEMAS})
SEMANTIC_CHECKPOINT_SCHEMAS = frozenset({ROUNDTRIP_CHECKPOINT_SCHEMA, COPY_CHECKPOINT_SCHEMA})
ACTIVE_STATUSES = frozenset({"feature_advice", "semantic_candidate_advice"})


class _InstructionScopeRejected(ValueError):
    """A valid historical candidate is outside the current inference scope."""

    def __init__(self, scope_report, checkpoint_descriptor):
        super().__init__("instruction is outside the declared single-action scope")
        self.scope_report = scope_report
        self.checkpoint_descriptor = checkpoint_descriptor


def _assess_scope(instruction):
    from ipfs_datasets_py.logic.intent_ir.formalize.instruction_scope import assess_intent_instruction_scope

    return assess_intent_instruction_scope(instruction)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _validate_descriptor_shape(descriptor):
    if descriptor is None:
        return
    if (type(descriptor) is not dict or set(descriptor) != {"schema", "path", "sha256"}
            or descriptor["schema"] not in {"intent-projection-feature-checkpoint/v1", *SEMANTIC_CHECKPOINT_SCHEMAS}
            or type(descriptor["path"]) is not str or not 0 < len(descriptor["path"]) <= 4096
            or type(descriptor["sha256"]) is not str or len(descriptor["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in descriptor["sha256"])
            or len(_encoded(descriptor)) > MAX_DESCRIPTOR_BYTES):
        raise ValueError("closed bounded Intent checkpoint descriptor required")


def _base(instruction, *, status, report=None, error_type=None, elapsed_ns=0, checkpoint_descriptor=None,
          scope_report=None, schema=SCHEMA):
    raw = instruction.encode("utf-8")
    value = {"schema": schema, "instruction_sha256": hashlib.sha256(raw).hexdigest(),
        "instruction_bytes": len(raw), "status": status, "report": report,
        "checkpoint_descriptor": checkpoint_descriptor,
        "error_type": error_type, "elapsed_ns": elapsed_ns,
        "continue_planning": True, "raw_instruction_preserved": True,
        "authority": "unverified_candidate_only", **{key: False for key in AUTHORITY_FIELDS}}
    if schema == SCHEMA:
        if scope_report is None and (
                type(report) is dict and report.get("schema") in SEMANTIC_REPORT_SCHEMAS
                or type(checkpoint_descriptor) is dict
                and checkpoint_descriptor.get("schema") in SEMANTIC_CHECKPOINT_SCHEMAS):
            scope_report = _assess_scope(instruction)
        value["scope_report"] = scope_report
    value["advice_sha256"] = hashlib.sha256(_encoded(value)).hexdigest()
    return value


def _scope_fallback(instruction, *, scope_report, checkpoint_descriptor=None, elapsed_ns=0):
    if (type(checkpoint_descriptor) is not dict
            or checkpoint_descriptor.get("schema") not in SEMANTIC_CHECKPOINT_SCHEMAS):
        checkpoint_descriptor = None
    else:
        try:
            _validate_descriptor_shape(checkpoint_descriptor)
        except ValueError:
            checkpoint_descriptor = None
    return _base(instruction, status="fail_open_instruction_scope", scope_report=scope_report,
        checkpoint_descriptor=checkpoint_descriptor, elapsed_ns=elapsed_ns)


def _native_report_validation(report, *, instruction, checkpoint_descriptor):
    """Dispatch only installed domain adapters; metadata never selects imports."""
    selected_schema = checkpoint_descriptor.get("schema") if type(checkpoint_descriptor) is dict else None
    report_schema = report.get("schema")
    if ((selected_schema == COPY_CHECKPOINT_SCHEMA and report_schema != COPY_REPORT_SCHEMA)
            or (report_schema == COPY_REPORT_SCHEMA and selected_schema not in {None, COPY_CHECKPOINT_SCHEMA})
            or (selected_schema == ROUNDTRIP_CHECKPOINT_SCHEMA and report_schema not in
                {ROUNDTRIP_REPORT_SCHEMA, *EXTENDED_REPORT_SCHEMAS})):
        raise ValueError("intent report and checkpoint schemas differ")
    if report_schema == COPY_REPORT_SCHEMA:
        from ipfs_datasets_py.logic.intent_ir.formalize.copy_roundtrip import validate_copy_intent_report

        return validate_copy_intent_report(report, instruction=instruction,
            checkpoint_descriptor=checkpoint_descriptor)
    if report.get("schema") in EXTENDED_REPORT_SCHEMAS:
        from ipfs_datasets_py.logic.intent_ir.formalize.extended_preplanning import validate_extended_intent_report

        return validate_extended_intent_report(report, instruction=instruction,
            checkpoint_descriptor=checkpoint_descriptor)
    if report.get("schema") == ROUNDTRIP_REPORT_SCHEMA:
        from ipfs_datasets_py.logic.intent_ir.formalize.roundtrip import validate_roundtrip_intent_report

        return validate_roundtrip_intent_report(report, instruction=instruction,
            checkpoint_descriptor=checkpoint_descriptor)
    from ipfs_datasets_py.logic.intent_ir.formalize.preplanning import validate_intent_instruction_report

    return validate_intent_instruction_report(report, instruction=instruction,
        checkpoint_descriptor=checkpoint_descriptor if report.get("status") == "feature_advice" else None)


def prepare_intent_advice(*, instruction: str, checkpoint_descriptor: dict | None = None,
                         checkpoint_descriptor_path: Path | None = None,
                         projection_request: dict | None = None,
                         projection_request_path: Path | None = None,
                         projection_request_sha256: str | None = None,
                         enabled: bool = True) -> dict:
    """Try the optional native frontend once; record ordinary failures openly.

    Cancellation and process termination are not swallowed. No downloads,
    provider calls, training, model discovery, or admission changes occur here.
    """
    if type(instruction) is not str or not instruction.strip():
        raise ValueError("the exact nonempty authorized instruction is required")
    if type(enabled) is not bool:
        raise ValueError("intent preprocessing selection must be boolean")
    if not enabled:
        return _base(instruction, status="disabled")
    started = time.monotonic_ns()
    try:
        if checkpoint_descriptor_path is not None:
            if checkpoint_descriptor is not None:
                raise ValueError("select one intent checkpoint descriptor")
            path = Path(checkpoint_descriptor_path)
            if path.is_symlink() or not path.is_file():
                raise ValueError("intent checkpoint descriptor must be a regular file")
            with path.open("rb") as stream:
                raw = stream.read(MAX_DESCRIPTOR_BYTES + 1)
            if len(raw) > MAX_DESCRIPTOR_BYTES:
                raise ValueError("intent checkpoint descriptor exceeds its byte bound")
            checkpoint_descriptor = json.loads(raw)
        _validate_descriptor_shape(checkpoint_descriptor)
        scope_report = None
        if type(checkpoint_descriptor) is dict and checkpoint_descriptor["schema"] in SEMANTIC_CHECKPOINT_SCHEMAS:
            scope_report = _assess_scope(instruction)
            if not scope_report["eligible_for_inference"]:
                return _scope_fallback(instruction, scope_report=scope_report,
                    checkpoint_descriptor=checkpoint_descriptor, elapsed_ns=time.monotonic_ns() - started)
        selected_request = projection_request
        try:
            if projection_request_path is not None:
                from ipfs_datasets_py.logic.intent_ir.formalize.projection_request import MAX_REQUEST_BYTES
                if (projection_request is not None or type(projection_request_sha256) is not str
                        or len(projection_request_sha256) != 64
                        or any(c not in "0123456789abcdef" for c in projection_request_sha256)):
                    raise ValueError("select one projection request with its exact file digest")
                request_path = Path(projection_request_path)
                if request_path.is_symlink() or not request_path.is_file():
                    raise ValueError("projection request must be a regular file")
                with request_path.open("rb") as stream:
                    raw = stream.read(MAX_REQUEST_BYTES + 1)
                if len(raw) > MAX_REQUEST_BYTES or hashlib.sha256(raw).hexdigest() != projection_request_sha256:
                    raise ValueError("projection request file digest or bound differs")
                selected_request = json.loads(raw)
                if selected_request is None:
                    raise ValueError("projection request must contain an envelope")
            elif projection_request_sha256 is not None:
                raise ValueError("file digest requires a projection request path")
        except Exception as exc:
            # Pass an intentionally invalid inert request to the optional layer.
            # It records failure while retaining independently valid base advice.
            selected_request = {"transport_error_category": type(exc).__name__}
        if type(checkpoint_descriptor) is dict and checkpoint_descriptor.get("schema") == COPY_CHECKPOINT_SCHEMA:
            from ipfs_datasets_py.logic.intent_ir.formalize.copy_roundtrip import (
                prepare_copy_intent_instruction as prepare_intent_instruction,
            )
        elif type(checkpoint_descriptor) is dict and checkpoint_descriptor.get("schema") == ROUNDTRIP_CHECKPOINT_SCHEMA:
            from ipfs_datasets_py.logic.intent_ir.formalize.extended_preplanning import (
                prepare_extended_intent_instruction as prepare_intent_instruction,
            )
        else:
            if selected_request is not None:
                raise ValueError("projection context requires a semantic roundtrip checkpoint")
            from ipfs_datasets_py.logic.intent_ir.formalize.preplanning import prepare_intent_instruction
        options = {} if selected_request is None else {"projection_request": selected_request}
        report = prepare_intent_instruction(instruction,
            checkpoint_descriptor=checkpoint_descriptor, **options)
        if type(report) is not dict or len(_encoded(report)) > MAX_REPORT_BYTES:
            raise ValueError("intent preprocessing report exceeds its closed boundary")
        if selected_request is not None and report.get("status") == "semantic_candidate_advice":
            if report.get("schema") not in {CONTEXT_REPORT_SCHEMA, COPY_REPORT_SCHEMA}:
                raise ValueError("selected projection request was silently dropped")
            applied = report.get("projection_request")
            if applied is not None and applied != selected_request:
                raise ValueError("preprocessing substituted a different projection request")
            if applied is None and report.get("extension_status") != "fail_open_projection_error":
                raise ValueError("selected projection request lacks an explicit disposition")
        _native_report_validation(report, instruction=instruction, checkpoint_descriptor=checkpoint_descriptor)
        return _base(instruction, status=report["status"], report=report,
            elapsed_ns=time.monotonic_ns() - started,
            checkpoint_descriptor=checkpoint_descriptor if report["status"] in ACTIVE_STATUSES
                or report.get("schema") in SEMANTIC_REPORT_SCHEMAS else None,
            scope_report=scope_report)
    except Exception as exc:
        return _base(instruction, status="fail_open_optional_error", error_type=type(exc).__name__,
            elapsed_ns=time.monotonic_ns() - started)


def validate_intent_advice(advice: dict, *, instruction: str) -> dict:
    """Validate the persisted sidecar, including native source/formula replay."""
    if type(advice) is not dict or advice.get("schema") not in {LEGACY_SCHEMA, SCHEMA}:
        raise ValueError("intent advice schema differs")
    expected = _base(instruction, status="disabled", schema=advice["schema"])
    if type(advice) is not dict or set(advice) != set(expected):
        raise ValueError("intent advice has unexpected fields")
    for key in ("schema", "instruction_sha256", "instruction_bytes", "continue_planning",
                "raw_instruction_preserved", "authority", *AUTHORITY_FIELDS):
        if type(advice[key]) is not type(expected[key]) or advice[key] != expected[key]:
            raise ValueError("intent advice source or authority binding differs")
    payload = {key: value for key, value in advice.items() if key != "advice_sha256"}
    if hashlib.sha256(_encoded(payload)).hexdigest() != advice["advice_sha256"]:
        raise ValueError("intent advice digest differs")
    if type(advice["elapsed_ns"]) is not int or advice["elapsed_ns"] < 0:
        raise ValueError("intent advice timing differs")
    _validate_descriptor_shape(advice["checkpoint_descriptor"])
    selected_schema = (advice["checkpoint_descriptor"] or {}).get("schema")
    semantic = (selected_schema in SEMANTIC_CHECKPOINT_SCHEMAS or
        type(advice["report"]) is dict and advice["report"].get("schema") in SEMANTIC_REPORT_SCHEMAS)
    scope_report = advice.get("scope_report")
    if advice["schema"] == SCHEMA:
        if scope_report is not None:
            from ipfs_datasets_py.logic.intent_ir.formalize.instruction_scope import validate_instruction_scope_report

            validate_instruction_scope_report(scope_report, instruction=instruction)
            if not semantic and advice["status"] != "fail_open_instruction_scope":
                raise ValueError("scope assessment requires semantic intent advice")
        elif semantic or advice["status"] == "fail_open_instruction_scope":
            raise ValueError("semantic intent advice requires its scope assessment")
    elif semantic:
        # Historical reports remain replayable, but cannot bypass current
        # eligibility by omitting the new envelope field.
        scope_report = _assess_scope(instruction)
    if (scope_report is not None and not scope_report["eligible_for_inference"]
            and advice["status"] in ACTIVE_STATUSES):
        raise _InstructionScopeRejected(scope_report, advice["checkpoint_descriptor"])
    if advice["report"] is None:
        scoped_fallback = advice["schema"] == SCHEMA and advice["status"] == "fail_open_instruction_scope"
        if scoped_fallback:
            if (scope_report is None or scope_report["eligible_for_inference"]
                    or advice["error_type"] is not None
                    or selected_schema not in {None, *SEMANTIC_CHECKPOINT_SCHEMAS}):
                raise ValueError("scope fallback requires a replayable unsupported instruction")
        elif advice["checkpoint_descriptor"] is not None:
            raise ValueError("fallback advice cannot select a checkpoint")
        if not scoped_fallback and advice["status"] not in {"disabled", "fail_open_optional_error", "fail_open_invalid_sidecar",
                                   "fail_open_advice_over_budget", "fail_open_advice_rejected"}:
            raise ValueError("intent advice lacks its native report")
        if advice["error_type"] is not None and (
                type(advice["error_type"]) is not str or not advice["error_type"].isidentifier()
                or len(advice["error_type"]) > 128):
            raise ValueError("intent advice error category differs")
    else:
        if scope_report is not None and not scope_report["eligible_for_inference"]:
            raise _InstructionScopeRejected(scope_report, advice["checkpoint_descriptor"])
        if len(_encoded(advice["report"])) > MAX_REPORT_BYTES or advice["error_type"] is not None:
            raise ValueError("intent advice report bounds differ")
        _native_report_validation(advice["report"], instruction=instruction,
            checkpoint_descriptor=advice["checkpoint_descriptor"])
        if advice["status"] != advice["report"]["status"]:
            raise ValueError("intent advice status differs")
        if advice["report"].get("schema") in SEMANTIC_REPORT_SCHEMAS:
            if advice["status"] == "semantic_candidate_advice":
                descriptor = advice["checkpoint_descriptor"]
                expected_schema = COPY_CHECKPOINT_SCHEMA if advice["report"]["schema"] == COPY_REPORT_SCHEMA else ROUNDTRIP_CHECKPOINT_SCHEMA
                if (type(descriptor) is not dict or descriptor.get("schema") != expected_schema
                        or descriptor.get("sha256") != advice["report"]["checkpoint_sha256"]):
                    raise ValueError("roundtrip inference selected a different checkpoint")
        elif advice["status"] == "feature_advice":
            from ipfs_datasets_py.logic.intent_ir.formalize.preplanning import load_intent_feature_checkpoint
            loaded = load_intent_feature_checkpoint(advice["checkpoint_descriptor"])
            if loaded["descriptor"]["sha256"] != advice["report"]["learned"]["checkpoint_sha256"]:
                raise ValueError("intent inference selected a different checkpoint")
        elif advice["checkpoint_descriptor"] is not None:
            raise ValueError("fallback advice cannot select a checkpoint")
    return advice


def load_intent_advice(*, path: Path, expected_sha256: str, instruction: str) -> dict:
    """A missing or changed advisory artifact drops advice, preserving the task."""
    try:
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("intent advice artifact is not a regular file")
        with path.open("rb") as stream:
            raw = stream.read(MAX_REPORT_BYTES + 16_385)
        if len(raw) > MAX_REPORT_BYTES + 16_384 or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("intent advice artifact changed")
        return validate_intent_advice(json.loads(raw), instruction=instruction)
    except _InstructionScopeRejected as exc:
        return _scope_fallback(instruction, scope_report=exc.scope_report,
            checkpoint_descriptor=exc.checkpoint_descriptor)
    except Exception as exc:
        return _base(instruction, status="fail_open_invalid_sidecar", error_type=type(exc).__name__)


def intent_planner_summary(advice: dict, *, instruction: str, maximum_bytes: int = 8192):
    """Return bounded, explicitly non-authoritative context or an omission record."""
    try:
        validate_intent_advice(advice, instruction=instruction)
        if advice["report"] is None or advice["status"] not in ACTIVE_STATUSES:
            return None, advice
        report = advice["report"]
        learned = report["learned"]
        semantic = report.get("schema") in SEMANTIC_REPORT_SCHEMAS
        targets = report["projections"] if semantic else report["deterministic"]["targets"]
        # Select only typed route metadata. Never duplicate natural language,
        # latent vectors, or reconstructed feature arrays in the model prompt.
        projections = [{"projection_id": row["projection_id"], "logic_family": row["logic_family"],
            "native_formula_count": len(row["native_formulas"]),
            "formula_kinds": sorted({formula["expression"]["kind"] for formula in row["native_formulas"]})}
            for row in targets["projections"]]
        summary = {"kind": "intent_instruction_advisory", "authority": "unverified_candidate_only",
            "instruction_sha256": advice["instruction_sha256"], "advice_sha256": advice["advice_sha256"],
            "notice": "Candidate planning context only. Preserve the original instruction and all independently admitted constraints.",
            "status": report["status"], "report_sha256": report["report_sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"] if semantic else learned["checkpoint_sha256"],
            "source_semantics_verified": False, "projections": projections,
            "gaps": report["gaps"], **{key: False for key in AUTHORITY_FIELDS}}
        if semantic:
            frame = learned["frame"]
            if (type(frame) is not dict or set(frame) != {"actor", "action", "object", "modality"}
                    or any(type(value) is not str or not 0 < len(value) <= 160 for value in frame.values())
                    or len(_encoded(frame)) > 4096):
                raise ValueError("roundtrip frame exceeds its planning context bound")
            summary.update(mode="semantic_roundtrip_candidate", frame=frame,
                learned_frame_generated=True,
                formula_producer="native_intent_compiler_from_learned_frame",
                reconstruction_is_proof=False)
            scope = advice.get("scope_report") or _assess_scope(instruction)
            summary["instruction_scope"] = {"policy_id": scope["policy_id"],
                "report_sha256": scope["report_sha256"], "grammar_shape": scope["grammar_shape"],
                "complete_consumption": scope["complete_consumption"],
                "scope_acceptance_is_semantic_verification": False}
            if report.get("schema") in {*EXTENDED_REPORT_SCHEMAS, COPY_REPORT_SCHEMA}:
                extended = report["extended_projections"]
                summary["extension_status"] = report["extension_status"]
                summary["extended_families"] = [] if extended is None else [
                    {"family": row["family_id"], "profile": row["profile_id"], "status": row["status"],
                     "projection_sha256": row["projection_sha256"], "semantics": row["semantics"],
                     "validation": [{"validator": check["validator"], "status": check["status"]}
                                    for check in row["validation"]],
                     "unsupported_count": len(row["unsupported"])} for row in extended["projections"]]
                summary["external_provers_executed"] = False
                for row in [] if extended is None else extended["projections"]:
                    if row["family_id"] != "transition_system" or row["profile_id"] is not None:
                        continue
                    for check in row["validation"]:
                        if check["validator"] != "finite_guarded_deadlock_scan":
                            continue
                        details = check["details"]
                        if (check["status"] not in {"passed", "failed"}
                                or type(details.get("deadlock_count")) is not int
                                or not 0 <= details["deadlock_count"] <= 128
                                or type(details.get("initial_valuation_count")) is not int
                                or not 1 <= details["initial_valuation_count"] <= 32
                                or details.get("all_configurations_enumerated") is not True
                                or details.get("premises_verified") is not False):
                            raise ValueError("guarded model diagnostics exceed the advisory contract")
                        summary["abstract_state_diagnostics"] = {"validator": check["validator"],
                            "status": check["status"], "deadlock_count": details["deadlock_count"],
                            "initial_valuation_count": details["initial_valuation_count"],
                            "all_configurations_enumerated": True, "premises_verified": False}
                if report.get("schema") in {CONTEXT_REPORT_SCHEMA, COPY_REPORT_SCHEMA}:
                    selected_request = report["projection_request"]
                    summary["projection_request_sha256"] = (
                        selected_request["request_sha256"] if selected_request is not None else None)
        else:
            summary.update(state_sha256=learned["inference"]["state_sha256"],
                decoded_formulas_generated=False, coverage=learned["inference"]["coverage"])
        encoded = _encoded(summary)
        if len(encoded) > maximum_bytes:
            return None, _base(instruction, status="fail_open_advice_over_budget")
        # A JSON string avoids increasing the planner's bounded mapping depth.
        return encoded.decode("utf-8"), advice
    except _InstructionScopeRejected as exc:
        return None, _scope_fallback(instruction, scope_report=exc.scope_report,
            checkpoint_descriptor=exc.checkpoint_descriptor)
    except Exception as exc:
        return None, _base(instruction, status="fail_open_advice_rejected", error_type=type(exc).__name__)


__all__ = ["prepare_intent_advice", "validate_intent_advice", "load_intent_advice", "intent_planner_summary"]
