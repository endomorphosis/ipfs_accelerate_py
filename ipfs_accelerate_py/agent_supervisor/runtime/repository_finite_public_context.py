"""Portable signed finite evidence for the worker, after semantic encoding.

This is read-only context. Public reconstruction checks the complete retained
artifacts without owner keys or new checker execution. The existing signed
handoff and native daemon still own edits, validation and publication.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

from . import repository_finite_handoff as handoff
from . import repository_finite_runner as runner
from . import local_planning_admission as local
from .doctor_candidate_runner import _directory, _read, _unique
from ..proof.formal_verification_contracts import content_identity

SCHEMA = "supervisor-finite-public-context@1"
MAX_BYTES = 2_000_000
NAMES = {"source": "captured_source.py", "compiled": "compiled.json", "driver": "driver.py",
    "request": "request.json", "tool_policy": "tool_policy.json", "trace": "observations.json",
    "python_process": "python_process.json", "lean_source": "FiniteInteger.lean",
    "lean_olean": "FiniteInteger.olean", "lean_certificate": "lean_certificate.json",
    "lean_process": "lean_process.json"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _raw(value):
    from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes
    return canonical_dag_json_bytes(value)


def _portable(observation):
    """Owner-side export only after native full-artifact reconstruction."""
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import validate_finite_integer_observation
    validate_finite_integer_observation(observation, expected_head=CodebaseHead.from_dict(observation["head"]),
        contract=IntegerOffsetContract.from_dict(observation["contract"]), inputs=observation["domain_inputs"],
        tool_policy=observation["tool_policy"])
    _require(observation["status"] == "observed" and set(observation["artifacts"]) == set(NAMES),
             "complete successful finite observations required")
    artifacts = {}
    for name, descriptor in observation["artifacts"].items():
        _require(Path(descriptor["path"]).name == NAMES[name], "unexpected native artifact name")
        with Path(descriptor["path"]).open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        _require(len(raw) <= MAX_BYTES and handoff._sha(raw) == descriptor["sha256"], "artifact changed during export")
        artifacts[name] = base64.b64encode(raw).decode("ascii")
    return {"observation": deepcopy(observation), "artifacts_base64": artifacts}


def _replay(value):
    """Rebase only file locations in a private integrity view; retain original CID.

    No file paths from the receipt are opened. Native validation sees our bounded
    copies. Rebased receipts are never published as checker execution evidence.
    """
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import validate_finite_integer_observation
    _require(type(value) is dict and set(value) == {"observation", "artifacts_base64"}, "closed portable observation required")
    observation, blobs = value["observation"], value["artifacts_base64"]
    _require(type(observation) is dict and observation["result_cid"] == cid_for_structured(
        {k: v for k, v in observation.items() if k != "result_cid"}), "original observation identity differs")
    _require(type(blobs) is dict and set(blobs) == set(NAMES) == set(observation["artifacts"]),
             "complete exact portable artifact population required")
    view, decoded = deepcopy(observation), {}
    with tempfile.TemporaryDirectory(prefix="finite-public-replay-") as directory:
        root = Path(directory).resolve()
        view["output"] = str(root)
        for name in sorted(NAMES):
            _require(type(blobs[name]) is str and len(blobs[name]) <= MAX_BYTES, "bounded artifact encoding required")
            raw = base64.b64decode(blobs[name], validate=True)
            descriptor = observation["artifacts"][name]
            _require(type(descriptor) is dict and set(descriptor) == {"path", "sha256", "size_bytes", "cid"}
                and type(descriptor["size_bytes"]) is int and len(raw) == descriptor["size_bytes"]
                and handoff._sha(raw) == descriptor["sha256"] and cid_for_bytes(raw) == descriptor["cid"]
                and Path(descriptor["path"]).name == NAMES[name], "portable artifact bytes differ")
            path = root / NAMES[name]
            path.write_bytes(raw)
            view["artifacts"][name]["path"] = str(path)
            decoded[name] = raw
        view["result_cid"] = cid_for_structured({k: v for k, v in view.items() if k != "result_cid"})
        (root / "result.json").write_bytes(_raw(view))
        validate_finite_integer_observation(view, expected_head=CodebaseHead.from_dict(observation["head"]),
            contract=IntegerOffsetContract.from_dict(observation["contract"]), inputs=observation["domain_inputs"],
            tool_policy=observation["tool_policy"])
    _require(observation["status"] == "observed", "successful original observation required")
    return observation, decoded


def _context(payload):
    _require(type(payload) is dict and set(payload) == {"schema", "handoff_sha256", "handoff", "instruction",
        "initial", "candidate", "artifact_cid", *handoff.FALSE} and payload["schema"] == SCHEMA
        and payload["artifact_cid"] == content_identity({k: v for k, v in payload.items() if k != "artifact_cid"})
        and all(payload[k] is False for k in handoff.FALSE), "closed public context identity required")
    signed = payload["handoff"]
    _require(type(signed) is dict and set(signed) == {"payload", "binding"}
        and handoff._sha(handoff._wire(signed)) == payload["handoff_sha256"], "original handoff bytes differ")
    original, binding = signed["payload"], signed["binding"]
    _require(set(binding) == {"identity", "profile_id", "signature"}, "closed original signature required")
    local.verify_did_key_signature(identity_did=binding["identity"], payload=original, signature=binding["signature"])
    _require(type(original) is dict and set(original) == runner.FIELDS and original["schema"] == handoff.SCHEMA
        and original["artifact_cid"] == content_identity({k: v for k, v in original.items() if k != "artifact_cid"})
        and all(original[k] is False for k in handoff.FALSE), "original handoff identity differs")
    instruction = payload["instruction"]
    _require(type(instruction) is str and handoff._sha(instruction.encode()) == original["instruction_sha256"],
             "complete original instruction differs")
    initial, before = _replay(payload["initial"])
    candidate, after = _replay(payload["candidate"])
    evidence = original["evidence"]
    from ..planning import finite_integer_codebase as matcher
    query = matcher.prepare_finite_integer_query(intent_document=matcher.build_finite_integer_intent(instruction), source_text=instruction)
    _require(query["supported"] and query["query_cid"] == evidence["query_cid"]
        and query["requirement_ids"] == evidence["complete_requirements"]
        and query["domain_inputs"] == evidence["inputs"] == initial["domain_inputs"] == candidate["domain_inputs"]
        and query["contract"] == initial["contract"] == candidate["contract"]
        and initial["result_cid"] == evidence["initial_observation_cid"]
        and candidate["result_cid"] == evidence["candidate_observation_cid"]
        and initial["head"] == evidence["source_head"] and initial["tool_policy"] == candidate["tool_policy"]
        and initial["source_sha256"] == original["edit"]["before_sha256"]
        and candidate["source_sha256"] == original["edit"]["after_sha256"]
        and after["source"] == base64.b64decode(original["edit"]["after_bytes_base64"], validate=True)
        and initial["type_clause_satisfied"] is True and initial["offset_clause_satisfied"] is False
        and initial["counterexample"] is not None and candidate["offset_clause_satisfied"] is True,
        "public source, instruction, residual or candidate lineage differs")
    compiled = json.loads(before["compiled"])
    return dict(schema=SCHEMA, original_instruction=instruction, source=before["source"].decode("ascii"),
        source_head=initial["head"], query=query, runtime_assumptions=compiled["assumptions"],
        finite_observations=initial["observations"], counterexample=initial["counterexample"],
        satisfied_requirements=[matcher.TYPE_STATEMENT_ID], residual_requirements=[matcher.OFFSET_STATEMENT_ID],
        initial_certificate=initial["lean_certificate"], candidate_certificate=candidate["lean_certificate"],
        initial_observation_cid=initial["result_cid"], candidate_observation_cid=candidate["result_cid"],
        scope=initial["scope"], replay_kind="public_artifact_integrity_only",
        checker_execution_performed=False, owner_keys_used=False, **handoff.FALSE)


def publish_finite_public_context(*, candidate, admission, instruction):
    """Package already checked owner evidence; no model/checker invocation."""
    declared = local.verify_local_benchmark_admission(admission, initial=True)
    original = candidate["signed_evidence"]
    _require(candidate["status"] == "candidate_ready" and local._verify_signature(original, declared["profile"]) == original["payload"]
        and original["payload"]["manifest_cid"] == declared["receipt"]["manifest_cid"], "independent admitted handoff required")
    payload = dict(schema=SCHEMA, handoff_sha256=candidate["handoff_sha256"], handoff=deepcopy(original),
        instruction=instruction, initial=_portable(candidate["preview"]["match"]["observation"]),
        candidate=_portable(candidate["candidate_observation"]), **handoff.FALSE)
    payload["artifact_cid"] = content_identity(payload)
    _context(payload)
    signed = local._signed(payload, declared["manifest"])
    artifact, sha = handoff._public_artifact(Path(declared["manifest"]["repository"]), signed)
    return dict(artifact=str(artifact), sha256=sha, artifact_cid=payload["artifact_cid"],
                handoff_sha256=candidate["handoff_sha256"])


def load_finite_public_context(*, artifact, expected_sha256, owner_did, profile_id, task_cid, handoff_sha256):
    artifact = Path(artifact).absolute()
    parent = _directory(artifact.parent)
    try:
        raw, metadata = _read(parent, artifact.name)
    finally:
        os.close(parent)
    _require(len(raw) <= MAX_BYTES and not stat.S_IMODE(metadata.st_mode) & 0o222
        and handoff._sha(raw) == expected_sha256, "immutable pinned public context required")
    signed = json.loads(raw, object_pairs_hook=_unique)
    _require(type(signed) is dict and set(signed) == {"payload", "binding"}, "closed signed context required")
    binding = signed["binding"]
    _require(type(binding) is dict and set(binding) == {"identity", "profile_id", "signature"}
        and binding["identity"] == owner_did and binding["profile_id"] == profile_id, "foreign context owner")
    local.verify_did_key_signature(identity_did=owner_did, payload=signed["payload"], signature=binding["signature"])
    payload = signed["payload"]
    _require(payload["handoff_sha256"] == handoff_sha256 and payload["handoff"]["binding"]["identity"] == owner_did
        and payload["handoff"]["binding"]["profile_id"] == profile_id
        and payload["handoff"]["payload"]["task_cid"] == task_cid, "context task or handoff differs")
    context = _context(payload)
    context["public_context_sha256"] = expected_sha256
    return context


def materialize_with_public_context(*, public_context, public_context_sha256, **options):
    context = load_finite_public_context(artifact=public_context, expected_sha256=public_context_sha256,
        owner_did=options["owner_did"], profile_id=options["profile_id"], task_cid=options["task_cid"],
        handoff_sha256=options["expected_sha256"])
    # The native dispatcher has already expanded semantic/world context. Add the
    # full read-only evidence after that encoding, preserving its original prompt.
    prompt = options["prompt"] + "\n[Finite repository evidence context]\n" + json.dumps(context, sort_keys=True)
    result = runner.materialize_finite_candidate(**{**options, "prompt": prompt})
    result["public_context"] = dict(sha256=public_context_sha256, owner_keys_used=False,
        integrity_replayed=True, injected_after_semantic_encoding=True,
        original_prompt_sha256=handoff._sha(options["prompt"].encode()),
        expanded_prompt_sha256=handoff._sha(prompt.encode()),
        assumptions_count=len(context["runtime_assumptions"]),
        counterexample=context["counterexample"], residual_requirements=context["residual_requirements"],
        original_instruction_sha256=handoff._sha(context["original_instruction"].encode()),
        checker_execution_performed=False, **handoff.FALSE)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "sha256", "task-cid", "owner-did", "profile-id", "public-context", "public-context-sha256"):
        parser.add_argument("--" + name, required=True)
    args = vars(parser.parse_args())
    args["expected_sha256"] = args.pop("sha256")
    try:
        result = materialize_with_public_context(**args, prompt=sys.stdin.buffer.read(256001).decode(), workspace=Path.cwd())
    except Exception as error:
        print(json.dumps(dict(status="refused", error_type=type(error).__name__, completion_authority=False)), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
