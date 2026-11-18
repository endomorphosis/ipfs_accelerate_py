"""Measure the fixed byte-target export without granting eligibility.

The codec stays a 259-symbol byte target and is not a semantic-action
language. The example export keeps the model input to source and context,
leaves labels and embedding metadata separate, and fixes both eligibility
masks at zero. A mutated codec or a changed measurement returns None.
"""

from __future__ import annotations

import hashlib
import json

from .dependency_block_resolution import EXPECTED_BYTE_TARGET_PROBE
from .meaning_material_probe import join_byte_target_inputs


def measure_byte_target_eligibility_probe() -> dict[str, object] | None:
    """Return the fixed byte-target contract, or None when owners disagree."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_BYTE_TARGET_PROBE:
        return None
    return measured


def _wire(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.formalization.autoencoder import (
        legal_relational_decoder_example_v1 as example,
    )
    from ipfs_datasets_py.logic.formalization.autoencoder import (
        legal_relational_target_codec_v1 as codec_module,
    )

    bundle = join_byte_target_inputs("P")
    codec = codec_module.build_relational_target_codec_v1()
    codec_sha = codec_module.relational_target_codec_sha256(codec)
    if (
        codec["vocabulary_size"] != 259
        or codec["semantic_action_language"] is not False
        or codec["grammar_constrained_generation"] is not False
        or codec["source_binding"] != "unsupported_in_authored_profile"
        or codec["byte_offset"] != 3
        or codec["bos_id"] != 1
        or codec["eos_id"] != 2
    ):
        raise RuntimeError("fixed byte codec changed")
    request = {
        "schema": example.REQUEST_SCHEMA,
        "task_id": example.TASK_ID,
        "exposure_status": "exposed",
        "embedding_reference": None,
    }
    raw = _wire(request)
    report = example.prepare_relational_decoder_example_v1(
        bundle["join_report"],
        raw,
        expected_example_request_sha256=hashlib.sha256(raw).hexdigest(),
        expected_join_report_sha256=bundle["join_sha256"],
        join_replay_inputs=bundle["replay"],
        codec=codec,
        expected_codec_sha256=codec_sha,
    )
    replayed = example.validate_relational_decoder_example_v1(
        report,
        bundle["join_report"],
        raw,
        expected_report_sha256=example.relational_decoder_example_sha256(report),
        expected_example_request_sha256=hashlib.sha256(raw).hexdigest(),
        expected_join_report_sha256=bundle["join_sha256"],
        join_replay_inputs=bundle["replay"],
        codec=codec,
        expected_codec_sha256=codec_sha,
    )
    if replayed != report:
        raise RuntimeError("example export did not replay")
    mutated = dict(codec)
    mutated["semantic_action_language"] = True
    try:
        example.prepare_relational_decoder_example_v1(
            bundle["join_report"],
            raw,
            expected_example_request_sha256=hashlib.sha256(raw).hexdigest(),
            expected_join_report_sha256=bundle["join_sha256"],
            join_replay_inputs=bundle["replay"],
            codec=mutated,
            expected_codec_sha256=codec_sha,
        )
    except ValueError:
        mutation_refused = True
    else:
        raise RuntimeError("semantic-action codec mutation was accepted")
    authority = report["authority"]
    if any(authority.values()):
        raise RuntimeError("example authority flag was true")
    label = report["label"]
    if not isinstance(label, dict):
        raise RuntimeError("byte target label was absent")
    target_ids = label["encoding"]["target_ids"]
    canonical = label["encoding"]["canonical_request_raw_utf8"].encode("utf-8")
    if target_ids != [1, *(3 + byte for byte in canonical), 2]:
        raise RuntimeError("byte target frame changed")
    model_input = report["model_input"]
    if set(model_input) != {"source_text", "context"}:
        raise RuntimeError("model input was not source and context")
    limitations = report["limitations"]
    separate = (
        "The exact model input contains source/context only; labels, "
        "alternative declarations and embedding metadata are separate."
    )
    not_admission = (
        "A candidate byte target is not a reviewed meaning, semantic-action "
        "decoder or training-admission decision."
    )
    if separate not in limitations or not_admission not in limitations:
        raise RuntimeError("example limitations no longer separate eligibility")
    if (
        report["profile"] != "candidate-source-separated-relational-byte-target/v1"
        or report["status"] != "candidate_target_prepared"
        or report["split"] != "unassigned"
        or report["training_mask"] != 0
        or report["evaluation_gold_mask"] != 0
        or report["embedding_reference"] is not None
        or report["embedding_reference_declaration_only"] is not True
        or report["provenance"]["exposure_status_caller_declared"] is not True
    ):
        raise RuntimeError("eligibility masks or export profile changed")
    return {
        "ast_action_head_added": False,
        "byte_offset": 3,
        "codec_mutation_refused": mutation_refused,
        "codec_profile": codec["profile"],
        "eligibility_granted": False,
        "embedding_reference_declaration_only": True,
        "evaluation_gold_mask": 0,
        "evaluation_gold_qualified": False,
        "example_profile": report["profile"],
        "example_replayed": True,
        "example_status": "candidate_target_prepared",
        "exposure_status_caller_declared": True,
        "fitting_executed": False,
        "grammar_constrained_generation": False,
        "model_executed": False,
        "model_input_fields": ["context", "source_text"],
        "semantic_action_language": False,
        "separate_from_model_input": True,
        "source_binding": "unsupported_in_authored_profile",
        "split": "unassigned",
        "target_ends_with_eos": True,
        "target_starts_with_bos": True,
        "teacher_qualified": False,
        "training_executed": False,
        "training_gold_qualified": False,
        "training_mask": 0,
        "training_qualified": False,
        "vocabulary_size": 259,
    }
