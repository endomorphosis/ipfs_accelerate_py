"""Measure complete prompts through the real planning consumer with a stub.

All source, requirements and provider responses are explicitly authored. The
native index and proposal parser run; target code, providers and solvers do not.
Stateless follow-ups charge the complete reconstructed prompt on every call.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from .terminal_source_view_experiment import _render, local_token_counter

SCHEMA = "terminal-repository-planning-consumer-experiment@1"
# Native lexical selection also includes module symbols. A function token that
# differs from its module basename selects exactly the authored three function
# capsules, without silently treating a module capsule as a fourth function.
QUERY = "compute_answer answer_label answer_store"
QUESTION = "Which original source implements compute_answer and needs independent validation?"
TASK_ID = "RPC-001"


def _wire(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def authored_sources():
    return {
        "answer.py": "# Authored inspection fixture; never executed.\ndef compute_answer(value):\n    return value + 1\n",
        "display.py": "def answer_label(value):\n    return str(value)\n",
        "storage.py": "def answer_store(value):\n    return str(value)\n",
        "requirements.txt": "Inspect answer.py before proposing a change. Preserve its public interface and require independent validation.\n",
        "plan.md": "# Public roadmap\nInspect the original answer function. Structural capsules do not establish behavioral correctness.\n",
        "tasks.todo.md": "# Authored tasks\n\n## RPC-001 Inspect the answer function\n- Status: ready\n- Priority: P1\n- Track: runtime\n- Depends on:\n- Outputs: answer.py\n- Validation: python -m unittest\n- Acceptance: Preserve the public interface and require independent validation.\n",
    }


def authored_contract(workspace):
    """Build a real native ledger from an explicit reviewed interpretation."""
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
    from ipfs_datasets_py.logic.intent_ir.schema import (
        IntentIRDocument, IntentKind, IntentModality, IntentStatement, NodeGrounding,
        ReviewStatus, SourceRef, SourceSpan, StatementKind,
    )

    source = (workspace / "requirements.txt").read_text(encoding="utf-8")
    raw = source.encode("utf-8")
    digest = _sha(raw)
    source_ref = SourceRef(ref_id="public-requirement", source_uri="fixture:public-requirement",
        source_id=digest, source_revision=digest, content_sha256=digest,
        span=SourceSpan(0, len(source)), review_status=ReviewStatus.MACHINE_EXTRACTED)
    statement = IntentStatement(statement_id="inspect-answer", kind=StatementKind.GOAL,
        modality=IntentModality.REQUIRED, normalized_text=source.strip(),
        source_ref_ids=(source_ref.ref_id,), predicate="modify", arguments=("agent", "answer.py"),
        confidence=0, grounding=NodeGrounding.INFERRED, review_status=ReviewStatus.MACHINE_EXTRACTED)
    document = IntentIRDocument(document_id="reviewed-answer:" + digest,
        title="Explicitly authored reviewed requirement", intent_kind=IntentKind.DECLARATIVE,
        sources=(source_ref,), statements=(statement,))
    document.validate()
    report = {
        "schema": "intent-reviewed-source-report@1", "source_sha256": digest,
        "source_bytes": len(raw), "source_characters": len(source),
        "producer": {"name": "authored-planning-consumer-fixture", "revision": "1"},
        "interpretation_status": "reviewed_candidate",
        "units": [{"unit_id": "unit:inspect-answer", "start_char": 0, "end_char": len(source),
            "start_byte": 0, "end_byte": len(raw), "text": source, "sha256": digest,
            "disposition": "interpreted_candidate", "reason": "explicit_reviewed_fixture"}],
        "candidates": [{"unit_id": "unit:inspect-answer", "candidate_intent_ir": document.to_dict()}],
        "proof_authority": False, "execution_authority": False,
        "completion_authority": False, "source_semantics_verified": False,
    }
    report["report_sha256"] = _sha(_wire(report))
    ledger = build_intent_requirement_ledger(source, source_report=report,
        source_identity={"path": "requirements.txt", "revision": "authored:v1"})
    return {"schema": "intent-plan-requirement-contract@1", "source_path": "requirements.txt",
        "ledger": ledger, "requirements": [{"requirement_id": ledger["requirements"][0]["requirement_id"],
            "outputs": [{"path": "answer.py", "effect": "modify", "media_type": "text/x-python"}],
            "validation_keys": ["public-answer"], "dependency_requirement_ids": []}]}


def make_config(workspace, artifacts, session, *, counter=None):
    from ipfs_accelerate_py.agent_supervisor.planning.repository_planning_context import RepositoryPlanningContext
    from ipfs_accelerate_py.agent_supervisor.planning.task_proposal_router import (
        TaskProposalRouterConfig, build_task_proposal_prompt,
    )
    return TaskProposalRouterConfig(repo_root=workspace, task_board_path=workspace / "tasks.todo.md",
        task_header_prefix="## RPC-", plan_path=workspace / "plan.md", artifact_dir=artifacts,
        prompt_builder=lambda task, plan_text: build_task_proposal_prompt(task=task,
            plan_text=plan_text, intro="Prepare an inspection proposal for the authored task.",
            requested_outputs=("exact files to inspect", "independent validation")),
        context_max_input_tokens=65536, context_tokenizer=counter,
        repository_context=RepositoryPlanningContext(session=session, generation=session.generation,
            intent_contract=authored_contract(workspace), query=QUERY, unresolved_question=QUESTION,
            query_limit=3))


def scripted_proposal(prompt):
    """A schema-valid inert proposal, with exact bindings copied from input."""
    from ipfs_accelerate_py.agent_supervisor.planning.task_proposal_router import TASK_IMPLEMENTATION_PROPOSAL_SCHEMA
    body = json.loads(prompt)
    authority = body["authority"]["task"]
    paths = list(authority["allowed_paths"])
    return _wire({"schema": TASK_IMPLEMENTATION_PROPOSAL_SCHEMA, "proposal_version": "1",
        "task_id": body["goal"]["task_id"], "repository_tree_id": body["tree_id"],
        "context_id": authority["context_id"],
        "files": [{"path": path, "operation": "modify", "rationale_references": ["authored:inspection"]} for path in paths],
        "validation_plan": list(authority["validation_commands"]),
        "risks": ["Structural evidence does not prove the proposed change correct."],
        "authority_claims": {"allowed_paths": paths, "validation_commands_only": True,
            "proof_authoritative": False, "completion_authoritative": False}})


def _exercise(counter):
    from ipfs_accelerate_py.agent_supervisor.planning.task_proposal_router import run_task_proposal_router
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession

    with tempfile.TemporaryDirectory(prefix="repository-planning-benchmark-", dir="/tmp") as temporary:
        root = Path(temporary)
        workspace = root / "source"
        workspace.mkdir()
        for name, text in authored_sources().items():
            (workspace / name).write_text(text, encoding="utf-8")
        with RepositorySourceSession(workspace, root / "state", representation="original") as session:
            config = make_config(workspace, root / "artifacts", session, counter=counter)
            native_query = session.context_query(generation=session.generation, query=QUERY, limit=3)
            candidates = native_query["candidates"]
            all_ids = tuple("repository-symbol:" + item["symbol_id"] for item in candidates)
            selected = tuple("repository-symbol:" + item["symbol_id"] for item in candidates if item["path"] == "answer.py")
            if len(all_ids) != 3 or len(selected) != 1:
                raise ValueError("authored optional population differs")
            measured, decisions, mandatory = {}, [], []
            calls = []

            def stub(prompt, invocation):
                calls.append(prompt)
                return scripted_proposal(prompt)

            with patch.dict(os.environ, {"IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE": "off"}), patch(
                "ipfs_accelerate_py.agent_supervisor.todo_daemon.llm.call_llm_router", side_effect=stub
            ):
                for arm, expansion in (("references", ()), ("one_expanded", selected), ("all_expanded", all_ids)):
                    configured = replace(config, artifact_dir=root / ("artifacts-" + arm),
                        repository_context=replace(config.repository_context, expansion_reference_ids=expansion))
                    payload = run_task_proposal_router(configured, generate=True,
                        provider="authored-stub", model="authored-stub", max_new_tokens=1024)
                    prompt = calls[-1]
                    decoded = json.loads(prompt)
                    proposal = json.loads((configured.artifact_dir / "rpc-001-proposal.json").read_text())
                    decisions.append({key: proposal[key] for key in ("task_id", "files", "validation_plan", "authority_claims")})
                    mandatory.append({item["reference_id"]: item for item in decoded["evidence"]
                        if item.get("metadata", {}).get("required") is True})
                    measured[arm] = {"consumer_calls": 1, "prompt_bytes": len(prompt.encode("utf-8")),
                        "local_tokenizer_proxy_tokens": counter(prompt) if counter else None,
                        "consumer_prompt_bytes_match": payload["prompt_bytes"] == len(prompt.encode("utf-8")),
                        "consumer_prompt_tokens_match": payload.get("prompt_tokens") == (counter(prompt) if counter else None),
                        "full_context_present": all(key in decoded for key in ("goal", "authority", "scope", "acceptance", "evidence")),
                        "expanded_native_population": sum(item.get("metadata", {}).get("repository_capsule_expanded") is True for item in decoded["evidence"]),
                        "prompt_sha256": _sha(prompt)}
                # A real second stateless invocation must pay the entire prompt
                # again; a compiler delta is never substituted for this call.
                for arm, expansion in (("one_expanded_repeat", selected), ("all_expanded_repeat", all_ids)):
                    configured = replace(config, artifact_dir=root / ("artifacts-" + arm),
                        repository_context=replace(config.repository_context, expansion_reference_ids=expansion))
                    payload = run_task_proposal_router(configured, generate=True,
                        provider="authored-stub", model="authored-stub", max_new_tokens=1024)
                    prompt = calls[-1]
                    decoded = json.loads(prompt)
                    proposal = json.loads((configured.artifact_dir / "rpc-001-proposal.json").read_text())
                    decisions.append({key: proposal[key] for key in ("task_id", "files", "validation_plan", "authority_claims")})
                    mandatory.append({item["reference_id"]: item for item in decoded["evidence"]
                        if item.get("metadata", {}).get("required") is True})
                    measured[arm] = {"consumer_calls": 1, "prompt_bytes": len(prompt.encode("utf-8")),
                        "local_tokenizer_proxy_tokens": counter(prompt) if counter else None,
                        "consumer_prompt_bytes_match": payload["prompt_bytes"] == len(prompt.encode("utf-8")),
                        "consumer_prompt_tokens_match": payload.get("prompt_tokens") == (counter(prompt) if counter else None),
                        "full_context_present": all(key in decoded for key in ("goal", "authority", "scope", "acceptance", "evidence")),
                        "prompt_sha256": _sha(prompt)}

            sequences = {}
            for name, components in (("references_then_one_stateless", ("references", "one_expanded_repeat")),
                                     ("all_expanded_twice_stateless", ("all_expanded", "all_expanded_repeat"))):
                sequences[name] = {"components": list(components), "consumer_calls": 2,
                    "prompt_bytes": sum(measured[item]["prompt_bytes"] for item in components),
                    "local_tokenizer_proxy_tokens": sum(measured[item]["local_tokenizer_proxy_tokens"] for item in components) if counter else None}
            cheaper = sequences["references_then_one_stateless"]["local_tokenizer_proxy_tokens"]
            full = sequences["all_expanded_twice_stateless"]["local_tokenizer_proxy_tokens"]
            checks = {"all_consumer_accounting_matches": all(row["consumer_prompt_bytes_match"] and row["consumer_prompt_tokens_match"] for row in measured.values()),
                "mandatory_evidence_identical": bool(mandatory[0]) and all(row == mandatory[0] for row in mandatory[1:]),
                "scripted_proposals_identical": all(row == decisions[0] for row in decisions[1:]),
                "complete_capsules_sent": all(row["full_context_present"] for row in measured.values()),
                "repeated_inputs_identical": all(measured[arm]["prompt_sha256"] == measured[arm + "_repeat"]["prompt_sha256"] for arm in ("one_expanded", "all_expanded")),
                "expanded_populations_match": [measured[arm]["expanded_native_population"] for arm in ("references", "one_expanded", "all_expanded")] == [0, 1, 3],
                "five_stub_dispatches": len(calls) == 5}
            return {"qualified": all(checks.values()), "checks": checks, "arms": measured,
                "sequences": sequences, "stub_dispatches": len(calls),
                "native_query_population": {key: native_query[key] for key in ("population_count", "matching_count", "selected_count", "omitted_count", "truncated")},
                "stateless_sequence_token_reduction_percent": round((full - cheaper) * 100 / full, 4) if full else None}


def run_experiment(*, output, tokenizer_cache=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counter, tokenizer = local_token_counter(tokenizer_cache)
    try:
        result = _exercise(counter)
    except Exception as error:
        result = {"qualified": False, "error_type": type(error).__name__, "arms": {}, "sequences": {}}
    result.update(schema=SCHEMA, local_tokenizer=tokenizer, provider_calls=0, containers_started=0,
        target_code_executed=False, proof_search_executed=False, training_executed=False,
        provider_usage_tokens=None, official_reward=None, content_bearing_messages_retained=False,
        measurement_scope="complete actual task-proposal consumer prompts for each stateless stub dispatch",
        limitations=["Authored reviewed interpretation is validated; arbitrary intent extraction and source satisfaction are not established.",
            "Security and UI status remain explicitly unavailable in this fixture; no domain verification is inferred.",
            "The provider callback is a deterministic schema fixture; no model quality, task success or billed saving is measured.",
            "Follow-ups resend and charge the complete reconstructed context, with no assumed provider-side prefix retention.",
            "Expansion sets are explicitly authored; the comparator expands all three selected function capsules, while unmatched module capsules remain omitted.",
            "Local token counts omit provider-specific message envelopes and billing adjustments."])
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
