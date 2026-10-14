"""Losslessly reference an independently admitted verbatim instruction block.

This pure representation codec follows the ordinary semantic metadata view.
The existing instruction owner supplies the signed source/block binding; this
codec neither verifies signatures nor grants omission or dispatch authority.
"""
from __future__ import annotations

import re

from . import semantic_metadata_view as metadata


MODE = "common-bindings-instruction@1"
VIEW_SCHEMA = "supervisor-semantic-instruction-view@1"
RECEIPT_SCHEMA = "supervisor-semantic-instruction-view-receipt@1"
REFERENCE_SCHEMA = "supervisor-verbatim-public-instruction-reference@1"
INSTRUCTIONS = (
    "The raw_sources entry containing $public_instruction refers to the complete "
    "verbatim admitted source in the appended ORIGINAL PUBLIC TASK REQUIREMENTS "
    "block. Read that block as the source text at the retained path; all bytes "
    "remain present in the complete input. No proof or authority is added."
)
_PREFIX = (
    "\n\nOriginal public task requirements (verbatim admitted source):\n"
    "These requirements supplement the native task objective. The independently admitted output scope, "
    "execution policy and validation contract remain authoritative. This text grants no additional "
    "write scope, publication, proof or completion authority. Apply the workspace mapping above to source "
    "operations and retain requested logical paths in reports.\n"
    "--- BEGIN ORIGINAL PUBLIC TASK REQUIREMENTS ---\n"
)
_END = "\n--- END ORIGINAL PUBLIC TASK REQUIREMENTS ---\n"
_BINDINGS = frozenset({"task_id", "task_cid", "context_cid", "manifest_cid", "artifact_sha256",
    "source_path", "source_sha256", "source_bytes", "block_sha256", "block_bytes"})
_SCHEMAS = frozenset("supervisor-public-instruction-inclusion@" + str(i) for i in range(1, 6))

SemanticInstructionViewError = metadata.SemanticMetadataViewError


def _binding(receipt: dict) -> dict:
    if (type(receipt) is not dict or type(receipt.get("schema")) is not str
            or receipt["schema"] not in _SCHEMAS):
        raise SemanticInstructionViewError("verified public instruction receipt required")
    if (receipt.get("manifest_signature_verified") is not True or receipt.get("verbatim_utf8") is not True
            or receipt.get("semantic_minification_applied") is not False
            or receipt.get("completion_authority") is not False
            or receipt.get("scope_expansion_authority") is not False):
        raise SemanticInstructionViewError("public instruction representation or authority differs")
    return _validate_binding({key: receipt.get(key) for key in _BINDINGS})


def _validate_binding(binding: dict) -> dict:
    if type(binding) is not dict or set(binding) != _BINDINGS:
        raise SemanticInstructionViewError("exact instruction identity binding required")
    for name, value in binding.items():
        if name.endswith("_bytes"):
            maximum = 32_768 if name == "source_bytes" else metadata.MAX_BYTES
            if type(value) is not int or not 1 <= value <= maximum:
                raise SemanticInstructionViewError("invalid instruction byte count")
        elif name.endswith("sha256"):
            if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
                raise SemanticInstructionViewError("invalid instruction digest")
        else:
            metadata._text(value, name)
    path = binding["source_path"]
    if (path.startswith("/") or "\\" in path or "\0" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))):
        raise SemanticInstructionViewError("canonical relative instruction path required")
    return dict(binding)


def _source(block: str, binding: dict) -> str:
    if (type(block) is not str or metadata._byte_count(block) != binding["block_bytes"]
            or metadata._sha(block) != binding["block_sha256"]):
        raise SemanticInstructionViewError("verbatim instruction block differs")
    raw, prefix, ending = block.encode("utf-8"), _PREFIX.encode("utf-8"), _END.encode("utf-8")
    start, end = len(prefix), len(prefix) + binding["source_bytes"]
    if not raw.startswith(prefix) or raw[end:end + len(ending)] != ending:
        raise SemanticInstructionViewError("instruction block source boundary differs")
    try:
        source = raw[start:end].decode("utf-8")
    except UnicodeError as error:
        raise SemanticInstructionViewError("instruction source must be exact UTF-8") from error
    if not source.strip() or "\0" in source or metadata._sha(source) != binding["source_sha256"]:
        raise SemanticInstructionViewError("instruction source identity differs")
    return source


def _reference(binding: dict) -> dict:
    return {"$public_instruction": {"schema": REFERENCE_SCHEMA,
        **{key: binding[key] for key in ("source_path", "source_sha256", "source_bytes", "block_sha256")}}}


def project_instruction_view(provider_prompt: str, *, instruction_block: str,
                             instruction_receipt: dict) -> metadata.SemanticMetadataView:
    """Reference only the exact selected source, preserving every other field."""
    base = metadata.project_semantic_metadata_view(provider_prompt)
    binding = _binding(instruction_receipt)
    source = _source(instruction_block, binding)
    wire = metadata._parse(base.provider_prompt)
    semantic = wire["translated_semantic"]
    sources, manifest = semantic.get("raw_sources"), semantic["manifest"]
    path = binding["source_path"]
    if (semantic["task_id"] != binding["task_id"] or type(sources) is not dict
            or path not in sources or type(sources[path]) is not str or sources[path] != source
            or type(manifest.get(path)) is not dict
            or manifest[path].get("sha256") != binding["source_sha256"]):
        raise SemanticInstructionViewError("selected instruction must match the captured raw source and task")
    sources[path] = _reference(binding)
    wire.update(schema=VIEW_SCHEMA, instruction_instructions=INSTRUCTIONS)
    candidate = metadata._json(wire)
    if metadata._byte_count(candidate) > metadata.MAX_BYTES:
        raise SemanticInstructionViewError("instruction candidate exceeds text bound")
    receipt = {**base.receipt, "schema": RECEIPT_SCHEMA, "view_schema": VIEW_SCHEMA,
        "candidate_view_sha256": metadata._sha(candidate), "candidate_view_bytes": metadata._byte_count(candidate),
        "candidate_view_proxy_tokens": metadata._proxy(candidate),
        "base_receipt": base.receipt, "instruction_binding": binding}
    result = metadata.SemanticMetadataView(provider_prompt, candidate, metadata._json(receipt))
    if restore_instruction_view(provider_prompt=candidate, receipt=receipt,
                                instruction_block=instruction_block) != provider_prompt:
        raise SemanticInstructionViewError("instruction view did not round-trip exactly")
    return result


def restore_instruction_view(*, provider_prompt: str, receipt: dict, instruction_block: str) -> str:
    """Restore exact transport bytes against the separately retained block."""
    if type(receipt) is not dict:
        raise SemanticInstructionViewError("retained instruction view receipt required")
    try:
        retained = metadata._parse(metadata._json(receipt))
    except (TypeError, ValueError, RecursionError, UnicodeError) as error:
        raise SemanticInstructionViewError("invalid bounded instruction view receipt") from error
    if (set(retained) != metadata._RECEIPT_FIELDS | {"base_receipt", "instruction_binding"}
            or retained.get("schema") != RECEIPT_SCHEMA or retained.get("view_schema") != VIEW_SCHEMA):
        raise SemanticInstructionViewError("instruction receipt schema differs")
    base = metadata._receipt(retained["base_receipt"])
    overrides = {"schema", "view_schema", "candidate_view_sha256", "candidate_view_bytes", "candidate_view_proxy_tokens"}
    if any(metadata._json(retained[key]) != metadata._json(base[key])
           for key in metadata._RECEIPT_FIELDS - overrides):
        raise SemanticInstructionViewError("instruction receipt base bindings differ")
    binding = _validate_binding(retained["instruction_binding"])
    source = _source(instruction_block, binding)
    wire = metadata._parse(provider_prompt)
    if (type(wire) is not dict or set(wire) != metadata._VIEW_FIELDS | {"instruction_instructions"}
            or wire.get("schema") != VIEW_SCHEMA or wire.get("instruction_instructions") != INSTRUCTIONS
            or metadata._sha(provider_prompt) != retained["candidate_view_sha256"]
            or metadata._byte_count(provider_prompt) != retained["candidate_view_bytes"]
            or metadata._proxy(provider_prompt) != retained["candidate_view_proxy_tokens"]):
        raise SemanticInstructionViewError("instruction candidate differs from retained receipt")
    semantic = wire.get("translated_semantic")
    if (type(semantic) is not dict or semantic.get("task_id") != binding["task_id"]
            or type(semantic.get("raw_sources")) is not dict
            or semantic["raw_sources"].get(binding["source_path"]) != _reference(binding)):
        raise SemanticInstructionViewError("instruction source reference differs")
    semantic["raw_sources"][binding["source_path"]] = source
    del wire["instruction_instructions"]
    wire["schema"] = metadata.VIEW_SCHEMA
    return metadata.restore_semantic_metadata_view(provider_prompt=metadata._json(wire), receipt=base)


def select_instruction_view(*, view: metadata.SemanticMetadataView, native_complete_prompt: str,
                            candidate_complete_prompt: str, instruction_block: str,
                            token_counter: metadata.SemanticMetadataTokenCounter | None = None,
                            ) -> metadata.SemanticMetadataSelection:
    """Select only when the complete prompt including its verbatim block shrinks."""
    if type(view) is not metadata.SemanticMetadataView:
        raise SemanticInstructionViewError("typed instruction metadata candidate required")
    if restore_instruction_view(provider_prompt=view.provider_prompt, receipt=view.receipt,
                                instruction_block=instruction_block) != view.native_prompt:
        raise SemanticInstructionViewError("instruction candidate identity differs")
    for prompt in (native_complete_prompt, candidate_complete_prompt):
        if type(prompt) is not str or metadata._byte_count(prompt) > metadata.MAX_BYTES:
            raise SemanticInstructionViewError("complete instruction input exceeds text bound")
    if native_complete_prompt.count(view.native_prompt) != 1:
        raise SemanticInstructionViewError("complete native transport occurrence differs")
    prefix, suffix = native_complete_prompt.split(view.native_prompt)
    if (candidate_complete_prompt != prefix + view.provider_prompt + suffix
            or candidate_complete_prompt.count(view.provider_prompt) != 1
            or suffix.count(instruction_block) != 1):
        raise SemanticInstructionViewError("complete instruction input changed its literal wrappers or verbatim block")
    measurement = None
    smaller = metadata._byte_count(candidate_complete_prompt) < metadata._byte_count(native_complete_prompt)
    if token_counter is None:
        selected = smaller and metadata._proxy(candidate_complete_prompt) < metadata._proxy(native_complete_prompt)
        fallback = None if selected else "complete_input_not_smaller_under_bytes_and_proxy"
    else:
        measurement = metadata._counter_measurement(token_counter, native_complete_prompt, candidate_complete_prompt)
        if measurement["token_counter_status"] != "counted":
            selected = False
            fallback = "complete_input_token_counter_" + measurement["token_counter_status"]
        else:
            selected = smaller and measurement["candidate_complete_tokens"] < measurement["native_complete_tokens"]
            fallback = None if selected else "complete_input_not_smaller_under_bytes_and_token_counter"
    chosen = candidate_complete_prompt if selected else native_complete_prompt
    receipt = {"schema": metadata.SELECTION_SCHEMA, "requested_mode": MODE,
        "view_receipt_sha256": metadata._sha(view.receipt_json),
        "selected_mode": MODE if selected else "legacy", "fallback_reason": fallback,
        "token_proxy": metadata.TOKEN_PROXY, "candidate_only": True,
        **{arm + "_complete_" + field: value for arm, prompt in
           (("native", native_complete_prompt), ("candidate", candidate_complete_prompt), ("selected", chosen))
           for field, value in (("sha256", metadata._sha(prompt)), ("bytes", metadata._byte_count(prompt)),
                                ("proxy_tokens", metadata._proxy(prompt)))},
        **{field: False for field in metadata._FALSE_FIELDS}}
    if measurement is not None:
        receipt.update(schema=metadata.TOKEN_SELECTION_SCHEMA, **measurement,
            selection_basis="token_counter", actual_provider_usage_verified=False,
            provider_usage_reported=False, cli_scaffolding_measured=False,
            complete_run_token_savings_claimed=False,
            selected_complete_tokens=(measurement["candidate_complete_tokens"] if selected
                                      else measurement["native_complete_tokens"]))
    return metadata.SemanticMetadataSelection(chosen, receipt["selected_mode"], metadata._json(receipt))
