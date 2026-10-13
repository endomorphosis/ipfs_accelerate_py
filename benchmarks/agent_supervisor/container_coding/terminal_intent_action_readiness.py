"""Content-free, inference-free readiness for the optional shared Intent reader.

Asset selection, current loader eligibility and complete-source grammar scope
are separate observations. None establish a successful decode or a proof.
"""
from __future__ import annotations

import hashlib


def intent_action_readiness(instruction: str, *, manifest: dict, config: dict | None = None) -> dict:
    from .terminal_deployment import validate_intent_action_384_binding

    binding = validate_intent_action_384_binding(manifest)
    if type(instruction) is not str or len(instruction.encode("utf-8")) > 32768:
        raise ValueError("bounded public instruction required")
    result = {
        "schema": "terminal-intent-action-384-readiness@1",
        "selection": "not_selected" if binding is None else "selected",
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "instruction_bytes": len(instruction.encode("utf-8")),
        "checkpoint_sha256": None if binding is None else binding["checkpoint_sha256"],
        "embedding_revision": None if binding is None else binding["embedding_revision"],
        "embedding_asset_pins_validated": binding is not None,
        "checkpoint_load_eligibility": "not_checked",
        "checkpoint_check_scope": "not_checked",
        "checkpoint_compatibility": None,
        "input_scope": "not_checked", "input_grammar": None,
        "inference_readiness": "not_selected",
        "continue_planning": True, "raw_instruction_preserved": True,
        "model_inference_executed": False, "embedding_inference_executed": False,
        "provider_calls": 0, "download_calls": 0, "training_steps": 0,
        "proof_authority": False, "source_semantics_verified": False,
        "completion_authority": False, "execution_authority": False,
    }
    if binding is None:
        if config is not None:
            raise ValueError("host Intent384 config requires its selected archive binding")
        return result

    # Check the complete unchanged instruction. Do not split, paraphrase or use
    # parser-derived targets in place of learned predictions.
    try:
        from ipfs_datasets_py.logic.intent_ir.formalize import action_contracts as codec
    except (ImportError, OSError, RuntimeError):
        result["input_scope"] = "unavailable"
    else:
        result["input_grammar"] = codec.SCHEMA
        try:
            codec.parse_contract(instruction)
        except (ValueError, TypeError):
            result["input_scope"] = "out_of_scope"
        except (ImportError, OSError, RuntimeError):
            result["input_scope"] = "unavailable"
        else:
            result["input_scope"] = "in_scope"

    if config is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.intent_384_advisor import _config
        config = _config(config)
        if config["checkpoint_sha256"] != binding["checkpoint_sha256"]:
            raise ValueError("host Intent384 checkpoint differs from selected archive binding")
        result["checkpoint_check_scope"] = "explicit_local_checkpoint_current_runtime"
        try:
            from ipfs_datasets_py.logic.formalization.autoencoder.intent_action_checkpoint_384 import (
                load_intent_action_checkpoint_384,
            )
            _, receipt = load_intent_action_checkpoint_384(config["checkpoint_path"],
                expected_sha256=config["checkpoint_sha256"])
        except (ValueError, TypeError, KeyError, OSError, ImportError, RuntimeError):
            # Exception messages may contain local paths or source contents.
            result["checkpoint_load_eligibility"] = "unavailable"
        else:
            result["checkpoint_load_eligibility"] = "eligible"
            result["checkpoint_compatibility"] = receipt

    if result["input_scope"] != "in_scope":
        result["inference_readiness"] = "input_" + result["input_scope"]
    elif result["checkpoint_load_eligibility"] == "eligible":
        result["inference_readiness"] = "eligible_for_inference"
    else:
        result["inference_readiness"] = "checkpoint_" + result["checkpoint_load_eligibility"]
    return result
