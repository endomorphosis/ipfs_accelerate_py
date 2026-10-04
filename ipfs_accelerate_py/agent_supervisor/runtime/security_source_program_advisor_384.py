"""Explicit local consumer of datasets-owned 384D Security source inference.

Advice retains predictions and abstentions. It never trains, downloads,
changes task authority, executes source Python, or replaces the Doctor's
independent repair contracts. Program Lake builds check syntax/types. The
explicit v2 state profile additionally checks finite operational correspondence.
Raw source is the default embedding input. A guarded AST-normalized view
requires explicit selection and retains the datasets-owned hybrid provenance.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time

SCHEMA = "supervisor-security-source-program-384-advice/v1"
CONFIG_SCHEMA = "supervisor-security-source-program-384-config/v1"
STATE_CONFIG_SCHEMA = "supervisor-security-source-program-384-config/v2"
STATE_SCHEMA = "supervisor-security-source-program-384-advice/v2"
MAX_ROWS = 16
MAX_SOURCE_BYTES = 65_536
MAX_ADVICE_BYTES = 2_097_152
NORMALIZED_INPUT_PROFILE = "guarded-source-normalization-384/v1"
FALSE = dict(proof_authority=False, execution_authority=False, completion_authority=False,
    mutation_authority=False, source_semantics_verified=False, whole_program_semantics_verified=False,
    security_specification_inferred=False, claim_proved=False, default_model_promoted=False)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, limit):
    # Reuse the supervisor's regular-file, link and concurrent-change checks.
    from .security_autoencoder_advisor import _read as read
    return read(Path(path), limit)


def _path(value):
    if type(value) is not str or not 0 < len(value) <= 4096 or not Path(value).is_absolute():
        raise ValueError("explicit absolute local artifact path required")
    return value


def _config(selection):
    if isinstance(selection, (str, Path)):
        selection = json.loads(_read(Path(selection).absolute(), 32768))
    required = {"schema", "checkpoint_path", "checkpoint_sha256", "decoder", "embedding_snapshot_path", "lake"}
    if type(selection) is dict and selection.get("schema") == STATE_CONFIG_SCHEMA:
        required.add("finite_state_domains")
    if (type(selection) is not dict or not required <= set(selection)
            or not set(selection) <= required | {"input_view"}):
        raise ValueError("closed explicit source-program configuration required")
    selected = deepcopy(selection)
    if selected.get("input_view", "raw") not in {"raw", "guarded_ast_normalized"}:
        raise ValueError("known explicit source input view required")
    if selected["schema"] not in {CONFIG_SCHEMA, STATE_CONFIG_SCHEMA} or selected["decoder"] not in {"structured", "sequence_v2"}:
        raise ValueError("known Security source-program configuration required")
    _path(selected["checkpoint_path"])
    if type(selected["checkpoint_sha256"]) is not str or not re.fullmatch(r"[a-f0-9]{64}", selected["checkpoint_sha256"]):
        raise ValueError("independent exact Security checkpoint SHA256 required")
    if selected["embedding_snapshot_path"] is not None:
        _path(selected["embedding_snapshot_path"])
    lake = selected["lake"]
    if lake is not None:
        if type(lake) is not dict or set(lake) != {"executable", "timeout_seconds"}:
            raise ValueError("closed explicit Lake selection required")
        _path(lake["executable"])
        if type(lake["timeout_seconds"]) not in (int, float) or not 0 < lake["timeout_seconds"] <= 60:
            raise ValueError("Lake timeout must be at most 60 seconds")
    return selected


def _base(status):
    return dict(schema=SCHEMA, status=status, continue_planning=True,
        checkpoint_selection=None, source_hashes={}, input_bindings=[], inference=None, lake=None,
        input_view="raw", normalization_profile=None,
        training_steps=0, download_calls=0, provider_calls=0, source_executed=False,
        source_scope="complete supplied Python files; no implicit extraction or source rewriting",
        elapsed_seconds=0., **FALSE)


def prepare_security_source_program_advice(*, config=None, source_rows=()):
    """Consume exact checkpoint weights using closed, independently hashed rows.

    Each row contains exactly id/source_text/source_sha256. Missing optional
    configuration performs no imports or inference. Input/inference failures
    remain advisory; they never block the existing supervisor workflow.
    The optional input_view selector changes only the embedding input contract;
    qualification and Lake replay always receive the original source bytes.
    V2 requires explicit finite_state_domains and returns separate optional
    source_state advice; a state failure never replaces the decoded candidate.
    """
    result = _base("disabled" if config is None else "fail_open_unavailable")
    if config is None:
        return result
    started = time.monotonic()
    stage = "configuration"
    try:
        selection = _config(config)
        result["checkpoint_selection"] = selection
        result["input_view"] = selection.get("input_view", "raw")
        if selection["schema"] == STATE_CONFIG_SCHEMA:
            result.update(schema=STATE_SCHEMA, source_state=None)
        stage = "source_inputs"
        if type(source_rows) is not list or not 1 <= len(source_rows) <= MAX_ROWS:
            raise ValueError("bounded explicit source rows required")
        captured_rows = []
        for row in source_rows:
            if type(row) is not dict or set(row) != {"id", "source_text", "source_sha256"}:
                raise ValueError("closed source rows required; targets are forbidden")
            row = dict(row)
            if type(row["id"]) is not str or not 0 < len(row["id"]) <= 256:
                raise ValueError("bounded source identifier required")
            if type(row["source_text"]) is not str or not 0 < len(row["source_text"].encode()) <= MAX_SOURCE_BYTES:
                raise ValueError("bounded nonempty source text required")
            if row["source_sha256"] != _sha(row["source_text"].encode()):
                raise ValueError("source digest differs")
            captured_rows.append(row)
        # Loader and inference callbacks may retain or mutate caller-owned rows.
        # All later inputs and checks use this detached, validated capture.
        source_rows = captured_rows
        if len({row["id"] for row in source_rows}) != len(source_rows):
            raise ValueError("duplicate source identity")
        result["source_hashes"] = {row["id"]: row["source_sha256"] for row in source_rows}
        result["input_bindings"] = [dict(source_id=row["id"], inference_id="input-" + str(index),
            source_sha256=row["source_sha256"]) for index, row in enumerate(source_rows)]
        stage = "checkpoint_loading"
        from ipfs_datasets_py.logic.formalization.autoencoder.source_program_runtime_384 import (
            load_source_program_decoder_384, build_decoded_source_program_lake,
        )
        loader = load_source_program_decoder_384
        load_options = dict(expected_sha256=selection["checkpoint_sha256"], decoder=selection["decoder"])
        compatible_structured = selection["schema"] == STATE_CONFIG_SCHEMA and selection["decoder"] == "structured"
        if compatible_structured:
            from ipfs_datasets_py.logic.formalization.autoencoder.source_program_runtime_384_v2 import (
                load_source_program_decoder_384_v2,
            )
            loader = load_source_program_decoder_384_v2
            load_options["input_view"] = result["input_view"]
        elif result["input_view"] == "guarded_ast_normalized":
            from ipfs_datasets_py.logic.formalization.autoencoder.normalized_source_program_runtime_384 import (
                load_normalized_source_program_decoder_384,
            )
            loader = load_normalized_source_program_decoder_384
        runtime = loader(selection["checkpoint_path"], **load_options)
        result["runtime"] = runtime.describe()
        if compatible_structured:
            result["checkpoint_compatibility"] = _checkpoint_compatibility(result["runtime"], selection)
        if result["input_view"] == "guarded_ast_normalized":
            _normalized_provenance(result["runtime"])
            result["normalization_profile"] = result["runtime"]["hybrid_profile"]
        stage = "embedding_and_inference"
        inference = runtime.infer_texts([row["source_text"] for row in source_rows],
            snapshot_path=selection["embedding_snapshot_path"])
        if result["input_view"] == "guarded_ast_normalized":
            _normalized_provenance(inference)
        if compatible_structured:
            checked = _checkpoint_compatibility(inference, selection)
            if checked != result["checkpoint_compatibility"]:
                raise ValueError("checkpoint compatibility changed between loading and inference")
        # Independently check the returned source identities before persisting
        # or forwarding any attached native-program evidence.
        expected = {row["inference_id"]: row["source_sha256"] for row in result["input_bindings"]}
        if (inference.get("domain_id") != "security_ir"
                or inference.get("checkpoint_sha256") != selection["checkpoint_sha256"]
                or len(inference["rows"]) != len(expected)
                or {row["id"]: row["source_sha256"] for row in inference["rows"]} != expected):
            raise ValueError("datasets inference source/checkpoint identity differs")
        result["inference"] = inference
        qualified = sum(row.get("source_contract", {}).get("status") == "qualified" for row in inference["rows"])
        result.update(status="source_candidate_advice" if qualified else "fail_open_no_qualified_candidates",
            qualified_candidate_count=qualified, source_count=len(source_rows))
        if selection["schema"] == STATE_CONFIG_SCHEMA:
            # This optional operational model binds code bytes, not the task's
            # Intent prompt. Bad domains or unavailable native tooling retain
            # the independent original inference and fail open for planning.
            stage = "optional_source_state"
            try:
                from .security_source_state_advisor_384 import consume_source_state_advice
                result["source_state"] = consume_source_state_advice(
                    inference=inference, source_rows=source_rows, input_bindings=result["input_bindings"],
                    finite_state_domains=selection["finite_state_domains"], lake=selection["lake"],
                    maximum_bytes=max(1024, min(1_048_576, MAX_ADVICE_BYTES - len(_wire(result)) - 256)))
            except Exception as error:
                result["source_state"] = dict(status="fail_open_unavailable", continue_planning=True,
                    error_type=type(error).__name__, source_executed=False, **FALSE)
        if selection["lake"] is not None:
            stage = "optional_lake"
            gate_rows = [dict(id="input-" + str(index), source_text=row["source_text"])
                         for index, row in enumerate(source_rows)]
            execution = build_decoded_source_program_lake(inference, gate_rows,
                lake_executable=selection["lake"]["executable"],
                timeout_seconds=selection["lake"]["timeout_seconds"])
            result["lake"] = execution.to_dict()
        stage = "advice_serialization"
        if result.get("source_state") is not None and len(_wire(result)) > MAX_ADVICE_BYTES:
            # Optional state output must not evict independently valid decoder
            # advice when combined with an existing program-Lake receipt.
            result["source_state"] = dict(status="fail_open_advice_over_budget", continue_planning=True,
                source_executed=False, **FALSE)
        if len(_wire(result)) > MAX_ADVICE_BYTES:
            raise ValueError("source program advice exceeds explicit bound")
    except Exception as error:
        if stage == "optional_lake" and result["inference"] is not None:
            result["lake"] = dict(status="unavailable", backend_executed=False,
                error_type=type(error).__name__, **FALSE)
        else:
            result.update(status="fail_open_unavailable", failure_stage=stage,
                error_type=type(error).__name__, inference=None, lake=None)
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def _normalized_provenance(report):
    """Require the selected hybrid profile to remain visible at both boundaries."""
    if (report.get("input_view") != "guarded_ast_normalized"
            or report.get("hybrid_profile") != NORMALIZED_INPUT_PROFILE
            or report.get("target_dependent_normalization") is not False
            or report.get("prediction_repair_performed") is not False):
        raise ValueError("datasets normalization provenance differs from explicit selection")


def _checkpoint_compatibility(report, selection):
    """Replay the narrow datasets-owned exception against exact artifact bytes."""
    from ipfs_datasets_py.logic.formalization.autoencoder.source_program_runtime_384_v2 import (
        verify_checkpoint_compatibility,
    )
    receipt = report.get("checkpoint_compatibility")
    if type(receipt) is not dict:
        raise ValueError("explicit shared checkpoint compatibility receipt required")
    checked = verify_checkpoint_compatibility(receipt, selection["checkpoint_path"], selection["checkpoint_sha256"])
    if checked != receipt:
        raise ValueError("checkpoint compatibility receipt differs from artifact replay")
    return deepcopy(checked)


def prepare_repository_source_program_advice(*, repository, paths, config, output=None):
    """Capture only explicitly permitted files, then recheck them after inference."""
    if config is None:
        return _base("disabled")
    result = _base("fail_open_source_capture")
    try:
        from ..analysis.planning_analysis_factory import _contains_secret, _credential_path_reason
        root = Path(repository).resolve(strict=True)
        if not isinstance(paths, (list, tuple)) or not 1 <= len(paths) <= 256 or len(set(paths)) != len(paths):
            raise ValueError("bounded explicit task source paths required")
        selected, unsupported, captured = [], [], {}
        for name in sorted(paths):
            if (type(name) is not str or Path(name).is_absolute() or Path(name).as_posix() != name
                    or ".." in Path(name).parts or _credential_path_reason(name)):
                raise ValueError("canonical screened source path required")
            if not name.endswith(".py"):
                unsupported.append(dict(id=name, reason="not_python_source"))
                continue
            raw = _read(root / name, MAX_SOURCE_BYTES)
            if _contains_secret(raw):
                raise ValueError("secret-like source refused")
            captured[name] = raw
            selected.append(dict(id=name, source_text=raw.decode("utf-8"), source_sha256=_sha(raw)))
        if selected:
            result = prepare_security_source_program_advice(config=config, source_rows=selected)
        else:
            result.update(status="fail_open_no_python_sources")
        result["unsupported_paths"] = unsupported
        if any(_read(root / name, MAX_SOURCE_BYTES) != raw for name, raw in captured.items()):
            raise ValueError("source changed during optional inference")
    except Exception as error:
        result = {**_base("fail_open_source_capture"), "error_type": type(error).__name__}
    if output is not None:
        try:
            artifact = Path(output).absolute()
            if artifact.resolve() != artifact or artifact.exists():
                raise ValueError("fresh canonical advice artifact required")
            raw = _wire(result)
            if len(raw) > MAX_ADVICE_BYTES:
                raise ValueError("source program advice exceeds explicit bound")
            with artifact.open("xb") as stream:
                stream.write(raw)
            return {**result, "artifact": str(artifact), "artifact_sha256": _sha(raw)}
        except Exception as error:
            return {**result, "persistence_status": "fail_open_unavailable", "persistence_error_type": type(error).__name__}
    return result


__all__ = ["prepare_security_source_program_advice", "prepare_repository_source_program_advice"]
