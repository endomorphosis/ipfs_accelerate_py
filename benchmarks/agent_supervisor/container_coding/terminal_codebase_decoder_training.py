"""Bounded, source-pinned native v1 production-head experiment.

Parser productions are supervised training targets, never inference fallback.
This adapter trains a new head over genuine frozen inherited lexical weights;
an optional older decoder supplies audited split history, not copied head weights.
The three roles are development controls, not a blind generalization estimate.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import warnings

SCHEMA = "terminal-codebase-decoder-training@1"
CORPUS_SCHEMA = "terminal-codebase-decoder-corpus@1"
PUBLIC_BOTTLE_SHA256 = "761756ce31753e526c48d28ccbca13a5d2493b16fe37aff3e1e4d2efaf3a2bba"
ROLE_POLICY = {"_hkey": "train", "_hval": "validation", "html_escape": "test"}
EPOCHS = 160
SEED = 1729
_AUTHORITY = {"proof_authority": False, "execution_authority": False,
              "completion_authority": False, "source_semantics_verified": False,
              "official_reward_measured": False, "whole_program_semantics_verified": False}


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_wire(value))


def _apis():
    from ipfs_datasets_py.logic.formalization.autoencoder.security import (
        security_formula_decoder as decoder, security_formula_grammar as grammar,
        security_formula_grammar_v2 as grammar_v2,
    )
    return decoder, grammar, grammar_v2


def _source(source_bytes, source_path, expected_source_sha256):
    from ipfs_datasets_py.logic.formalization.autoencoder.source_screening import (
        SourceSecretError, _contains_secret, _credential_path_reason,
    )
    if (type(source_bytes) is not bytes or not 0 < len(source_bytes) <= 1_048_576
            or type(source_path) is not str or not source_path or len(source_path) > 512
            or PurePosixPath(source_path).is_absolute() or ".." in PurePosixPath(source_path).parts
            or "\\" in source_path or not source_path.endswith(".py")):
        raise ValueError("bounded exact public Python source and relative path required")
    if _sha(source_bytes) != expected_source_sha256:
        raise ValueError("complete public source identity differs")
    if _credential_path_reason(source_path) or _contains_secret(source_bytes):
        raise SourceSecretError("decoder input refused by shared source screen")
    if any(byte in source_bytes for byte in (b"\r", b"\v", b"\f", b"\x00")):
        raise ValueError("exact LF source mapping required")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source_bytes.decode("utf-8"), type_comments=True)
    if sum(1 for _ in ast.walk(tree)) > 65_536:
        raise ValueError("complete AST bound exceeded; source is not truncated")
    return tree


def validate_terminal_decoder_split_history(*, samples, parent_checkpoint=None):
    """Refuse source or alpha/literal shape roles that contradict a genuine parent."""
    decoder, grammar, _ = _apis()
    prior = None
    if parent_checkpoint is not None:
        schema = parent_checkpoint.get("schema")
        if schema == decoder.SCHEMA:
            prior = decoder.load_security_formula_decoder(parent_checkpoint)
        elif schema == "security-formula-production-decoder@2":
            raise ValueError("v2 historical ancestor roles unavailable; this v1 experiment requires an independently complete v1 split history")
        elif schema == "security-formula-production-continuation@1":
            from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_decoder_continuation import load_security_formula_decoder_continuation
            prior = load_security_formula_decoder_continuation(parent_checkpoint)
            # The continuation kernel accepts v2 parents. Its merged history
            # cannot recover omitted v1 roles from a v2 parent's hash-only
            # transfer receipt, so this narrow experiment refuses that lineage.
            ancestor = prior["training"]["parent_descriptor"]
            seen = set()
            for _ in range(8):
                digest = _sha(_wire(ancestor))
                if digest in seen:
                    raise ValueError("cyclic historical decoder lineage")
                seen.add(digest)
                if ancestor.get("schema") == decoder.SCHEMA:
                    decoder.load_security_formula_decoder(ancestor)
                    break
                if ancestor.get("schema") != "security-formula-production-continuation@1":
                    raise ValueError("continuation historical ancestor roles incomplete for this v1 experiment")
                ancestor = load_security_formula_decoder_continuation(ancestor)["training"]["parent_descriptor"]
            else:
                raise ValueError("bounded historical decoder lineage exceeded")
        else:
            raise ValueError("explicit supported historical decoder version required")
    history, matches = {}, []
    if prior is not None:
        for role, rows in prior["training"].get("split_history", prior["training"]["splits"]).items():
            for row in rows:
                for kind, field in (("body", "source_sha256"), ("alpha_literal_shape", "program_shape_sha256")):
                    identity = (kind, row[field])
                    if identity in history and history[identity] != role:
                        raise ValueError("historical source/shape split contradiction")
                    history[identity] = role
    current = {}
    for sample in samples:
        raw = sample["source"].encode()
        if _sha(raw) != sample["source_sha256"]:
            raise ValueError("declared training source digest differs")
        for kind, digest in (("body", _sha(raw)), ("alpha_literal_shape", grammar.source_shape(raw))):
            identity = (kind, digest)
            if identity in current and current[identity] != sample["split"]:
                raise ValueError("current source/shape cross-split overlap")
            current[identity] = sample["split"]
            if identity in history:
                if history[identity] != sample["split"]:
                    raise ValueError("historical source/shape role reassignment refused before fitting")
                matches.append({"sample_id": sample["id"], "kind": kind, "sha256": digest,
                                "role": sample["split"]})
    return {"parent_checkpoint": deepcopy(parent_checkpoint),
            "parent_training_receipt_sha256": _sha(_wire(prior["training"])) if prior else None,
            "historical_identity_count": len(history), "same_role_matches": matches,
            "cross_role_matches": [], "checked_before_fit": True,
            "parent_head_weights_copied": False, "parent_quality_requalified": False}


def prepare_terminal_codebase_decoder_corpus(*, source_bytes, source_path,
        parent_checkpoint=None, expected_source_sha256=PUBLIC_BOTTLE_SHA256):
    """Observe all functions and independently report both frozen grammar frontiers."""
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_corpus import _qualified_functions, _verify_span
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formalization_evaluation import _function_span
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_logic_v2 import lower_pure_function_v2, UnsupportedLogic
    decoder, grammar, grammar_v2 = _apis()
    tree = _source(source_bytes, source_path, expected_source_sha256)
    found = _qualified_functions(tree)
    if len(found) > 1024:
        raise ValueError("complete function bound exceeded; source is not truncated")
    functions, samples = [], []
    for node, qualified, enclosing in found:
        body, binding = _function_span(source_bytes, node)
        _verify_span(source_bytes, body, binding)
        original_ast = ast.dump(node, include_attributes=False)
        identity = {"path": source_path, "source_sha256": _sha(source_bytes),
                    "qualified_name": qualified, "source_binding": binding}
        row = {"unit_id": _sha(_wire(identity)), "qualified_name": qualified,
            "symbol": node.name, "enclosing_scope": enclosing, "source_path": source_path,
            "source_sha256": _sha(source_bytes), "normalized_body_sha256": _sha(body),
            "source_ast_sha256": _sha(original_ast.encode()), "source_binding": binding,
            "line": node.lineno, "end_line": node.end_lineno, "split": None,
            "eligibility": {}, "normalized_source": None, "teacher_nodes": [],
            "teacher_targets_used_for_fit": False}
        for label, parser in (("v1", grammar), ("v2", grammar_v2)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    parsed = parser.parse_formula_source(body)
                extracted = ast.parse(body.decode(), type_comments=True).body
                if len(extracted) != 1 or ast.dump(extracted[0], include_attributes=False) != original_ast:
                    raise ValueError("normalized function AST differs from original")
                eligibility = {"status": "supported", "reason": None,
                    "program_shape_sha256": parser.source_shape(body), "node_count": len(parsed["nodes"])}
            except (parser.UnsupportedFormulaSource, SyntaxError, ValueError, UnicodeError, RecursionError) as exc:
                eligibility = {"status": "unsupported", "reason": str(exc),
                               "program_shape_sha256": None, "node_count": 0}
                parsed = None
            row["eligibility"][label] = eligibility
            if label == "v1" and parsed is not None:
                row["normalized_source"] = body.decode()
                row["teacher_nodes"] = parsed["nodes"]
        try:
            if row["eligibility"]["v2"]["status"] != "supported":
                raise UnsupportedLogic("v2_grammar_unsupported")
            lower_pure_function_v2(body)
            typed = {"status": "supported", "reason": None}
        except (UnsupportedLogic, ValueError) as exc:
            typed = {"status": "unsupported", "reason": str(exc)}
        row["eligibility"]["typed_v2_logic"] = typed
        if not enclosing and node.name in ROLE_POLICY and row["eligibility"]["v1"]["status"] == "supported":
            row["split"] = ROLE_POLICY[node.name]
            row["teacher_targets_used_for_fit"] = row["split"] == "train"
            samples.append({"id": "terminal-v1:" + row["unit_id"], "split": row["split"],
                "source": row["normalized_source"], "source_sha256": row["normalized_body_sha256"]})
        functions.append(row)
    if len({row["unit_id"] for row in functions}) != len(functions):
        raise ValueError("duplicate complete source unit identity")
    lineage = validate_terminal_decoder_split_history(samples=samples, parent_checkpoint=parent_checkpoint)
    eligible = [row for row in functions if row["eligibility"]["v1"]["status"] == "supported"]
    ready = len(samples) == len(ROLE_POLICY) and len(eligible) == len(samples) and all(
        sum(sample["split"] == role for sample in samples) == 1 for role in ROLE_POLICY.values())
    value = {"schema": CORPUS_SCHEMA, "source_path": source_path,
        "source_sha256": _sha(source_bytes), "source_bytes": len(source_bytes),
        "role_policy": dict(ROLE_POLICY), "functions": functions, "samples": samples,
        "counts": {"functions_observed": len(functions), "v1_supported": len(eligible),
            "v1_unsupported": len(functions) - len(eligible),
            "v2_supported": sum(row["eligibility"]["v2"]["status"] == "supported" for row in functions),
            "typed_v2_logic_supported": sum(row["eligibility"]["typed_v2_logic"]["status"] == "supported" for row in functions),
            "selected_samples": len(samples)}, "lineage": lineage,
        "training_input_contract_satisfied": ready,
        "teacher_target_origin": "exact admitted source AST parser productions",
        "heldout_scope": "predeclared development controls excluded from gradients; no blind generalization claim",
        "complete_module_semantics_supported": False, "training_steps": 0,
        "implementation_sha256": {"adapter": _sha(Path(__file__).read_bytes()),
            "v1_grammar": _sha(Path(grammar.__file__).read_bytes()),
            "v2_grammar": _sha(Path(grammar_v2.__file__).read_bytes()),
            "span_adapter": _sha(Path(_function_span.__code__.co_filename).read_bytes())}, **_AUTHORITY}
    value["corpus_sha256"] = _sha(_wire(value))
    return value


def _candidate_rows(corpus, checkpoint):
    decoder, _, _ = _apis()
    candidates, controls = [], []
    for row in corpus["functions"]:
        if row["split"] is None:
            continue
        raw = row["normalized_source"].encode()
        inference = decoder.decode_security_formula(source_bytes=raw, checkpoint=checkpoint,
                                                    source_path=row["source_path"])
        candidates.append({key: row[key] for key in ("unit_id", "qualified_name", "symbol", "source_path",
            "source_sha256", "normalized_body_sha256", "source_binding", "source_ast_sha256", "split")}
            | {"candidate_source": inference["candidate_source"], "decode": inference,
               "inference_sha256": _sha(_wire(inference))})
        for name, options in (("model_off", {"model_enabled": False}),
                              ("zero_production_heads", {"weight_ablation": "zero_production_heads"})):
            result = decoder.decode_security_formula(source_bytes=raw, checkpoint=checkpoint,
                source_path=row["source_path"], **options)
            controls.append({"name": name, "unit_id": row["unit_id"], "decode": result,
                "inference_sha256": _sha(_wire(result)), "checkpoint_mutated": False,
                "candidate_emitted": result["candidate_source"] is not None})
    wrong = dict(checkpoint, weights_sha256="0" * 64)
    try:
        decoder.load_security_formula_decoder(wrong)
    except ValueError:
        controls.append({"name": "wrong_checkpoint", "status": "refused", "checkpoint_mutated": False})
    else:
        raise ValueError("wrong checkpoint was accepted")
    try:
        prepare_terminal_codebase_decoder_corpus(source_bytes=b"# wrong source\n", source_path=corpus["source_path"],
            parent_checkpoint=corpus["lineage"]["parent_checkpoint"], expected_source_sha256=corpus["source_sha256"])
    except ValueError:
        controls.append({"name": "wrong_source", "status": "refused", "source_mutated": False})
    else:
        raise ValueError("wrong source was accepted")
    return candidates, controls


def _report(*, output, policy, corpus, checkpoint, published_binding, weight_transfer):
    decoder, _, _ = _apis()
    from ipfs_datasets_py.logic.formalization.autoencoder.security.published_legal_initializer import validate_published_legal_initializer
    checked = validate_published_legal_initializer(expected_receipt=published_binding)
    if checked["initializer"] != weight_transfer:
        raise ValueError("published binding and inherited initializer differ")
    loaded = decoder.load_security_formula_decoder(checkpoint)
    training = loaded["training"]
    if (training["training_provenance_sha256"] != corpus["corpus_sha256"]
            or training["epochs"] != policy["epochs"] or training["seed"] != policy["seed"]
            or training["inherited_lexical_sha256"] != _sha(_wire(loaded["weights"]["lexical"]))
            or loaded["weights"]["lexical"]["initializer_sha256"] != weight_transfer["initializer_sha256"]
            or loaded["weights"]["lexical"]["published_source_pin"] != published_binding["source_pin"]
            or loaded["weights"]["lexical"] != decoder._initializer(weight_transfer, published_binding)):
        raise ValueError("actual training policy/corpus/lexical lineage differs")
    expected_splits = {role: [] for role in ("train", "validation", "test")}
    _, grammar, _ = _apis()
    for sample in corpus["samples"]:
        expected_splits[sample["split"]].append({"id": sample["id"], "source_sha256": sample["source_sha256"],
            "program_shape_sha256": grammar.source_shape(sample["source"].encode())})
    if training["splits"] != expected_splits:
        raise ValueError("fitted source inventory differs from frozen corpus")
    candidates, controls = _candidate_rows(corpus, checkpoint)
    value = {"schema": SCHEMA, "output": str(output), "policy": policy, "corpus": corpus,
        "checkpoint": checkpoint, "published_binding": published_binding, "weight_transfer": weight_transfer,
        "loaded_training_receipt": training, "candidates": candidates, "controls": controls,
        "learned_candidate_count": sum(row["candidate_source"] is not None for row in candidates),
        "native_complete_program_ir_count": sum(row["decode"]["validation"]["native_lowering_complete"] for row in candidates),
        "training_steps": training["training_steps"], "native_kernel_calls": training["native_kernel_calls"],
        "provider_calls": 0, "download_calls": 0, "executes_source": False,
        "source_AST_checks_are_runtime_semantics_proofs": False,
        "asymptotic_optimizer_convergence_proved": False,
        "loss_trace_scope": "native train-only cross entropy recorded before each optimizer update; frozen final CE measured separately",
        "parent_role": "genuine frozen published lexical rows; optional older decoder history only; no head continuation",
        **_AUTHORITY}
    value["report_sha256"] = _sha(_wire(value))
    return deepcopy(value)


def _policy(corpus, parent_checkpoint, published_binding, weight_transfer):
    return {"schema": "terminal-codebase-decoder-frozen-policy@1", "epochs": EPOCHS, "seed": SEED,
        "latent_width": 32, "role_policy": dict(ROLE_POLICY), "corpus_sha256": corpus["corpus_sha256"],
        "source_sha256": corpus["source_sha256"], "parent_checkpoint": deepcopy(parent_checkpoint),
        "published_binding_sha256": _sha(_wire(published_binding)),
        "initializer_sha256": weight_transfer["initializer_sha256"],
        "training_data_scope": "caller_declared_development_controls",
        "objective": "native family_logits train-only production cross entropy",
        "teacher_scope": "parser-derived productions only from train _hkey; validation/test labels evaluated after fitting",
        "checkpoint_selection": "final checkpoint after fixed budget; no selection by heldout score",
        "success_selected_retries": 0, "frozen_before_fit": True,
        "implementation_sha256": _sha(Path(__file__).read_bytes()), **_AUTHORITY}


def train_terminal_codebase_decoder(*, source_bytes, source_path, output,
        parent_checkpoint=None, weight_transfer=None, published_binding=None):
    """Freeze the fixed 160-step policy, then perform one native fit without retries."""
    decoder, _, _ = _apis()
    from ipfs_datasets_py.logic.formalization.autoencoder.security import security_autoencoder_checkpoint as portable
    from ipfs_datasets_py.logic.formalization.autoencoder.security.published_legal_initializer import validate_published_legal_initializer
    if published_binding is None:
        raise ValueError("exact existing published initializer binding required; no downloads or fabricated seed")
    admitted = validate_published_legal_initializer(expected_receipt=published_binding)
    if weight_transfer is None:
        weight_transfer = admitted["initializer"]
    if weight_transfer != admitted["initializer"]:
        raise ValueError("selected genuine initializer differs")
    corpus = prepare_terminal_codebase_decoder_corpus(source_bytes=source_bytes, source_path=source_path,
                                                      parent_checkpoint=parent_checkpoint)
    if not corpus["training_input_contract_satisfied"]:
        raise ValueError("fixed three-unit development corpus is not ready")
    excluded = [Path(weight_transfer["output"]), Path(published_binding["output"])]
    if parent_checkpoint is not None:
        excluded.append(Path(parent_checkpoint["output"]))
    output = portable._namespace(Path(output), fresh=True, excluded=tuple(excluded))
    policy = _policy(corpus, parent_checkpoint, published_binding, weight_transfer)
    output.mkdir(parents=True, mode=0o700)
    _write(output / "policy.json", policy)
    _write(output / "corpus.json", corpus)
    checkpoint = decoder.train_security_formula_decoder(samples=corpus["samples"], weight_transfer=weight_transfer,
        published_binding=published_binding, output=output / "checkpoint", epochs=EPOCHS, seed=SEED,
        latent_width=32, training_data_scope=policy["training_data_scope"],
        training_provenance_sha256=corpus["corpus_sha256"])
    rebuilt = prepare_terminal_codebase_decoder_corpus(source_bytes=source_bytes, source_path=source_path,
                                                       parent_checkpoint=parent_checkpoint)
    if rebuilt != corpus:
        raise ValueError("source inventory or historical lineage changed during fitting")
    report = _report(output=output, policy=policy, corpus=corpus, checkpoint=checkpoint,
                     published_binding=published_binding, weight_transfer=weight_transfer)
    _write(output / "decoder-report.json", report)
    validate_terminal_codebase_decoder(expected=report, source_bytes=source_bytes, source_path=source_path)
    return report


def validate_terminal_codebase_decoder(*, expected, source_bytes, source_path):
    """Replay exact source, native package and raw inference; never fit weights."""
    from ipfs_datasets_py.logic.formalization.autoencoder.security import security_autoencoder_checkpoint as portable
    if type(expected) is not dict or expected.get("schema") != SCHEMA:
        raise ValueError("exact decoder experiment report required")
    output = portable._namespace(Path(expected["output"]))
    if {path.name for path in output.iterdir()} != {"checkpoint", "policy.json", "corpus.json", "decoder-report.json"}:
        raise ValueError("closed decoder experiment inventory required")
    if portable._read(output / "decoder-report.json", 16_777_216) != _wire(expected):
        raise ValueError("decoder experiment persisted report differs")
    policy = expected["policy"]
    if (policy["epochs"] != EPOCHS or policy["seed"] != SEED or policy["role_policy"] != ROLE_POLICY
            or policy["source_sha256"] != PUBLIC_BOTTLE_SHA256
            or policy["implementation_sha256"] != _sha(Path(__file__).read_bytes())
            or policy["frozen_before_fit"] is not True or policy["success_selected_retries"] != 0
            or expected["checkpoint"]["output"] != str(output / "checkpoint")
            or policy["published_binding_sha256"] != _sha(_wire(expected["published_binding"]))):
        raise ValueError("fixed experiment policy or checkpoint namespace differs")
    corpus = prepare_terminal_codebase_decoder_corpus(source_bytes=source_bytes, source_path=source_path,
        parent_checkpoint=policy["parent_checkpoint"], expected_source_sha256=policy["source_sha256"])
    if (corpus != expected["corpus"] or policy["corpus_sha256"] != corpus["corpus_sha256"]
            or not corpus["training_input_contract_satisfied"]
            or policy != _policy(corpus, policy["parent_checkpoint"], expected["published_binding"], expected["weight_transfer"])
            or portable._read(output / "policy.json", 1_048_576) != _wire(policy)
            or portable._read(output / "corpus.json", 16_777_216) != _wire(corpus)):
        raise ValueError("frozen corpus/policy differs from exact public source")
    actual = _report(output=output, policy=policy, corpus=corpus, checkpoint=expected["checkpoint"],
        published_binding=expected["published_binding"], weight_transfer=expected["weight_transfer"])
    if _wire(actual) != _wire(expected):
        raise ValueError("decoder source/model/raw candidate/control replay differs")
    return actual
