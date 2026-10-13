"""Authored context compilation, question expansion and diagnostic retry.

This is a local provider-input proxy over real native repository capsules. It
executes neither target code nor a model. Domain requirements are explicitly
authored declarations with unresolved statuses, not inferred satisfaction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from .terminal_source_view_experiment import _render, local_token_counter

SCHEMA = "terminal-repository-context-experiment@1"
ARMS = ("full_capsules_initial", "references_then_question")
QUERY = "invoice_total invoice_label invoice_storage"
QUESTION = "Which original source contains the invoice_total function that requires inspection?"
TARGET = "ledger.py"
REQUIREMENTS = {
    "intent": {
        "requirement_id": "authored:intent:invoice", "status": "reviewed_candidate",
        "required_text": "Inspect invoice_total before proposing a change. Preserve the existing public function signature and arithmetic behavior unless the reviewed task explicitly requires otherwise. All proposed changes must retain exact original-source bindings and require independent validation.",
        "source_satisfaction": "unresolved", "proof_authority": False,
    },
    "security": {
        "requirement_id": "authored:security:local", "status": "unassessed",
        "required_text": "Limit this authored investigation to the three declared local source files. Do not read credentials, contact external services, execute target code, or treat structural capsule confidence as permission to mutate source. Security compliance remains unassessed by this fixture.",
        "source_satisfaction": "unresolved", "execution_authority": False,
    },
    "ui": {
        "requirement_id": "authored:ui:labels", "status": "not_validated",
        "required_text": "Preserve existing display labels and public interface names while inspecting invoice arithmetic. This fixture has no rendered interface, interaction trace, or accessibility checker; UI and UX obligations must remain visible and unresolved throughout selection, expansion, and retry.",
        "source_satisfaction": "unresolved", "completion_authority": False,
    },
}
DIAGNOSTIC = {
    "schema": "authored-context-diagnostic@1", "status": "needs_original_source",
    "message": "The structural capsule identifies an inspection target. Exact original source and independent checks are still required before proposing a mutation.",
    "proof_authority": False, "execution_authority": False, "completion_authority": False,
}


def _wire(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def authored_sources():
    return {
        TARGET: b"# Authored arithmetic fixture; no target execution.\ndef invoice_total(subtotal, tax):\n    return subtotal + tax\n",
        "display.py": b"# Authored interface fixture.\ndef invoice_label(value):\n    return str(value)\n",
        "storage.py": b"# Authored storage fixture.\ndef invoice_storage(value):\n    return str(value)\n",
    }


def required_references():
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import build_text_context_references
    return tuple(reference for family, body in REQUIREMENTS.items()
        for reference in build_text_context_references(_wire(body),
            reference_prefix="authored-domain:" + family, kind="authored-domain-requirement",
            required=True, coverage_ids=(body["requirement_id"],)))


def _metric(text, counter):
    tokens, complete = None, False
    if counter is not None:
        try:
            tokens = counter(text)
            if type(tokens) is not int or tokens < 0:
                raise ValueError("invalid tokenizer count")
            complete = True
        except Exception:
            tokens = None
    return {"messages": int(bool(text)), "rendered_json_bytes": len(text.encode("utf-8")),
            "local_tokenizer_proxy_tokens": tokens, "token_accounting_complete": complete}


def _traffic(frames, counter):
    components = {name: _metric(value, counter) for name, value in frames.items()}
    complete = all(row["token_accounting_complete"] for row in components.values())
    total = {"messages": sum(row["messages"] for row in components.values()),
        "rendered_json_bytes": sum(row["rendered_json_bytes"] for row in components.values()),
        "local_tokenizer_proxy_tokens": sum(row["local_tokenizer_proxy_tokens"] for row in components.values()) if complete else None,
        "token_accounting_complete": complete}
    return {"total": total, "components": components, "wire_accounting_complete": True,
            "component_totals_match": all(total[key] == sum(row[key] for row in components.values())
                for key in ("messages", "rendered_json_bytes"))}


def _mandatory_intact(capsule, required):
    selected = {item.reference_id: item for item in capsule.evidence}
    return all(item.reference_id in selected and selected[item.reference_id].required
        and selected[item.reference_id].summary == item.summary
        and selected[item.reference_id].coverage_ids == item.coverage_ids for item in required)


def _decision(capsule, required, source):
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SemanticCapsule
    if not _mandatory_intact(capsule, required):
        raise ValueError("mandatory domain population changed")
    bodies = []
    for item in capsule.evidence:
        if item.path == TARGET and item.metadata.get("repository_capsule_expanded") is True:
            # The adapter defines this exact envelope. Admission references
            # intentionally share some CID fields with the native capsule;
            # they must never be guessed to be additional capsule bodies.
            envelope = json.loads(item.summary)
            if not isinstance(envelope, dict) or not isinstance(envelope.get("capsule"), dict):
                raise ValueError("expanded native capsule envelope is missing")
            bodies.append(SemanticCapsule.from_dict(envelope["capsule"]))
    bound = [item for item in bodies if item.source_cid == cid_for_bytes(source)]
    if len(bound) != 1:
        raise ValueError("selected native fact is missing or ambiguous")
    # This decision is the complete, explicit scripted fixture rule. It does
    # not claim model adoption, behavior verification, or benchmark reward.
    return {"decision": "inspect_exact_original_before_proposal", "target_path": TARGET,
        "stable_symbol_id": bound[0].stable_symbol_id, "source_cid": bound[0].source_cid,
        "required_body_sha256": sorted(_sha(item.summary) for item in required),
        "domain_statuses": {family: body["status"] for family, body in REQUIREMENTS.items()},
        "proof_authority": False, "execution_authority": False, "completion_authority": False}


def _exercise(counter):
    from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
        ContextCompiler, EvidenceExpansionRequest, render_context_capsule,
    )
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
        ContextBudget, ContextReference, canonical_context_json_bytes,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_context_compiler import RepositoryContextAdapter
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession

    sources, required = authored_sources(), required_references()
    with tempfile.TemporaryDirectory(prefix="repository-context-benchmark-", dir="/tmp") as temporary:
        root = Path(temporary)
        workspace = root / "source"
        workspace.mkdir()
        for name, raw in sources.items():
            (workspace / name).write_bytes(raw)
        with RepositorySourceSession(workspace, root / "state", representation="original") as session:
            compiler = ContextCompiler(ContextBudget(max_input_tokens=65536, max_items=64,
                max_item_bytes=32768, max_text_bytes=16384), tokenizer=counter)
            adapter = RepositoryContextAdapter(session, compiler)
            generation = session.status()["generation"]
            initial = adapter.compile(generation=generation, query=QUERY, unresolved_question=QUESTION,
                required_evidence=required, required_reference_ids=tuple(item.reference_id for item in required),
                required_coverage_ids=tuple(sorted(body["requirement_id"] for body in REQUIREMENTS.values())),
                objective_id="authored:invoice-inspection", objective_revision="authored:1",
                policy_id="authored:local-context", policy_revision="authored:1",
                goal={"objective": "Identify the source for bounded invoice inspection."},
                authority={"proof_authority": False, "execution_authority": False, "completion_authority": False},
                scope={"paths": sorted(sources)}, acceptance={"required_ids": sorted(body["requirement_id"] for body in REQUIREMENTS.values()),
                    "rule": "Preserve all domain obligations and return the same source-bound inspection decision."},
                limit=3, max_bytes=32768)
            handles = tuple(item for item in initial.capsule.evidence
                            if item.reference_id.startswith("repository-symbol:"))
            selected = tuple(item for item in handles if item.path == TARGET)
            if len(handles) != 3 or len(selected) != 1:
                raise ValueError("authored optional population differs")
            initial_text = render_context_capsule(initial.capsule)
            question = EvidenceExpansionRequest(QUESTION, (selected[0].reference_id,))
            question_text = _wire({"unresolved_question": question.unresolved_question,
                                   "reference_ids": list(question.reference_ids)})
            # Host-only materialization constructs the comparator input. Only
            # its complete rendered initial context is charged to that arm.
            all_expanded = adapter.expand_for_question(initial.capsule,
                EvidenceExpansionRequest(QUESTION, tuple(item.reference_id for item in handles)),
                generation=generation, max_bytes=65536)
            one_expanded = adapter.expand_for_question(initial.capsule, question,
                generation=generation, max_bytes=65536)
            full = all_expanded.reconstructed_capsule
            compact = one_expanded.reconstructed_capsule
            diagnostic = ContextReference(reference_id="authored:diagnostic", kind="authored-diagnostic",
                summary=_wire(DIAGNOSTIC), referenced_content_id="sha256:" + _sha(_wire(DIAGNOSTIC)),
                byte_count=len(_wire(DIAGNOSTIC).encode("utf-8")))
            arms = {}
            decisions = []
            for mode, parent in ((ARMS[0], full), (ARMS[1], compact)):
                retried = adapter.retry(parent, generation=generation, evidence=(*parent.evidence, diagnostic))
                final = retried.reconstructed_capsule
                decision = _decision(final, required, sources[TARGET])
                decisions.append(decision)
                frames = {
                    "initial_context": render_context_capsule(full) if mode == ARMS[0] else initial_text,
                    "question": question_text,
                    "question_expansion": "" if mode == ARMS[0] else canonical_context_json_bytes(one_expanded.delta_capsule.to_record()).decode("utf-8"),
                    "diagnostic_retry": canonical_context_json_bytes(retried.delta_capsule.to_record()).decode("utf-8"),
                }
                checks = {
                    "mandatory_population_complete": _mandatory_intact(initial.capsule, required) and _mandatory_intact(parent, required) and _mandatory_intact(final, required),
                    "three_optional_native_symbols": len(handles) == 3,
                    "model_facing_discovery_exposes_selected_handle": selected[0].reference_id in initial_text and TARGET in initial_text,
                    "selected_native_fact_verified": bool(decision["source_cid"]),
                    "required_statuses_unchanged": decision["domain_statuses"] == {family: body["status"] for family, body in REQUIREMENTS.items()},
                    "retry_preserves_invariant_core": retried.invariant_core_preserved,
                    "diagnostic_is_only_retry_transmission": tuple(item.reference_id for item in retried.delta_capsule.evidence) == (diagnostic.reference_id,),
                    "diagnostic_payload_identical": next(item.summary for item in retried.delta_capsule.evidence if item.reference_id == diagnostic.reference_id) == _wire(DIAGNOSTIC),
                    "authority_ceilings_preserved": all(decision[key] is False for key in ("proof_authority", "execution_authority", "completion_authority")),
                    "original_sources_unchanged": all((workspace / name).read_bytes() == raw for name, raw in sources.items()),
                }
                arms[mode] = {"qualified": all(checks.values()), "checks": checks, "traffic": _traffic(frames, counter),
                    "scripted_decision_sha256": _sha(_wire(decision)), "required_reference_count": len(required),
                    "initial_native_body_count": 3 if mode == ARMS[0] else 0,
                    "question_expanded_body_count": 0 if mode == ARMS[0] else 1,
                    "final_context_input_tokens": final.input_tokens, "content_bearing_messages_retained": False}
            paired = decisions[0] == decisions[1]
            unchanged = all((workspace / name).read_bytes() == raw for name, raw in sources.items())
        result = {"arms": arms, "paired_scripted_decisions_identical": paired,
                  "original_sources_unchanged": unchanged}
    result["private_fixture_removed"] = not root.exists()
    result["qualified"] = paired and unchanged and result["private_fixture_removed"] and all(arm["qualified"] for arm in arms.values())
    return result


def run_experiment(*, output, tokenizer_cache=None):
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("context experiment output must be a new canonical directory")
    counter, tokenizer = local_token_counter(tokenizer_cache)
    output.mkdir(parents=True)
    try:
        result = _exercise(counter)
    except Exception as error:
        result = {"qualified": False, "error_type": type(error).__name__, "arms": {},
                  "paired_scripted_decisions_identical": False}
    result.update(schema=SCHEMA, local_tokenizer=tokenizer, provider_calls=0, containers_started=0,
        target_code_executed=False, training_executed=False, proof_search_executed=False,
        provider_usage_tokens=None, official_reward=None, complete_multi_ir_qualified=False,
        content_bearing_messages_retained=False, required_domain_population=len(REQUIREMENTS),
        measurement_scope="canonical model-facing initial context, identical question, actual selected expansion delta, and diagnostic retry delta",
        limitations=["Domain constraints and their unresolved statuses are explicitly authored; no source satisfaction is inferred.",
            "The same deterministic fixture rule checks both arms; no model decision, task success, provider adoption or billed saving is established.",
            "Each arm charges its complete initial context once; no compiler receipt or host-only native reread is represented as model traffic.",
            "The full-body arm already has all three native capsules when the identical question arrives, so its expansion transmission is empty.",
            "Local tokenizer counts measure these canonical strings, without a provider-specific chat envelope or billing adjustments.",
            "The native structural capsules retain their own authority ceiling; this is not complete learned Codebase/Intent/Security/UI integration."])
    (output / "qualification.json").write_text(_render(result), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-cache", type=Path)
    args = parser.parse_args(argv)
    result = run_experiment(output=args.output, tokenizer_cache=args.tokenizer_cache)
    print(_render({"schema": SCHEMA, "qualified": result["qualified"], "output": str(args.output), "provider_calls": 0}), end="")
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
