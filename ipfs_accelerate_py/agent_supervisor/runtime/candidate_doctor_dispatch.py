"""Doctor constraints carried by an independently admitted coding continuation.

The owner policy bounds child attempts and outer router invocations. Its token
and cost observations are separate from the unchanged residual-proposal budget.
This module cannot mint an owner admission from serialized context.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .worker_source_views import canonical_json

SCHEMA = "supervisor-doctor-candidate-dispatch-observation@1"
PUBLIC_SCHEMA = "supervisor-doctor-candidate-dispatch-observation@2"
_OBSERVATION_FLAGS = dict(canonical_source_current=True, workspace_source_current=True,
    original_obligations_preserved=True, proposal_budget_changed=False,
    owner_round_number=2, native_token_cap_enforced=False, native_cost_cap_enforced=False,
    completion_authority=False, proof_authority=False)
_DIGEST_FIELDS = ("coding_policy_sha256", "doctor_context_sha256", "doctor_artifact_sha256",
    "doctor_advisory_sha256", "proposal_budget_sha256", "original_requirements_sha256",
    "source_manifest_sha256")


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _same(left, right):
    return canonical_json(left) == canonical_json(right)


def prepare_doctor_dispatch_projection(*, prompt, admission):
    """Retain an exact typed owner projection after its independent admission."""
    from .doctor_candidate_context import DoctorCandidateContext
    context = prompt._doctor_context
    _need(type(context) is DoctorCandidateContext
          and context._requirements is prompt._requirements
          and context._snapshot is prompt._snapshot,
          "typed Doctor context belongs to another candidate prompt")
    metadata = context.revalidate()
    original = prompt._requirements._doctor_requirements
    path, digest, _ = original._request
    value = dict(context=metadata,
        artifact=dict(path=path, sha256=digest, bytes=len(original.artifact_bytes)),
        advisory_sha256=_sha(context.advisory_bytes), advisory_bytes=len(context.advisory_bytes),
        coding_policy=admission["doctor_coding_policy"])
    snapshot = prompt._snapshot
    if getattr(snapshot, "_public_validation", None) is not None:
        observed = snapshot._public_validation.revalidate(launch=snapshot._validation_launch,
            retired_worker=snapshot._retired_worker, snapshot=snapshot)
        value["public_validation"] = dict(contract=snapshot._validation_launch.public_validation_contract.metadata,
            observation=observed)
    _need(admission["doctor_context_sha256"] == _sha(canonical_json(metadata))
        and admission["doctor_advisory_sha256"] == value["advisory_sha256"]
        and admission["doctor_advisory_bytes"] == value["advisory_bytes"],
        "Doctor handoff projection differs from owner admission")
    return value


def doctor_dispatch_observation(projection):
    """Reproduce wire facts after caller validation, without checking freshness.

Historical auditors may compare this expected observation with a retained
receipt. Calling this helper alone establishes neither freshness nor admission.
"""
    context, policy = projection["context"], projection["coding_policy"]
    original = context["original_doctor"]
    result = dict(schema=SCHEMA, policy_id=policy["policy_id"],
        coding_policy_sha256=_sha(canonical_json(policy)),
        doctor_context_sha256=_sha(canonical_json(context)),
        doctor_artifact_sha256=projection["artifact"]["sha256"],
        doctor_advisory_sha256=projection["advisory_sha256"],
        doctor_advisory_bytes=projection["advisory_bytes"],
        proposal_budget=dict(original["residual_budget"]),
        proposal_budget_sha256=_sha(canonical_json(original["residual_budget"])),
        task_cid=original["task_cid"], context_cid=original["context_cid"],
        native_capsule_id=original["native_capsule_id"],
        original_requirements_sha256=context["original_requirements_sha256"],
        source_manifest_sha256=original["source_manifest_sha256"],
        source_files=len(context["candidate_inventory_paths"])+len(context["unchanged_inventory_paths"]),
        **_OBSERVATION_FLAGS)
    if "public_validation" in projection:
        public = projection["public_validation"]
        result.update(schema=PUBLIC_SCHEMA,
            public_validation_contract_sha256=_sha(canonical_json(public["contract"])),
            public_validation_observation_sha256=_sha(canonical_json(public["observation"])),
            actual_prior_returncode=public["observation"]["actual_returncode"],
            public_validation_status=public["observation"]["status"])
    return result


def validate_doctor_dispatch_observation(value):
    """Check a receipt's closed wire shape; no source or admission is replayed."""
    from .source_formalization_context import _digest
    from .doctor_residual_budget import LIMIT_FIELDS, validate_residual_budget
    from .candidate_round_policy import DOCTOR_CODING_POLICY, PUBLIC_VALIDATION_POLICY
    fields = {"schema", "policy_id", "proposal_budget", "doctor_advisory_bytes", "source_files",
        "task_cid", "context_cid", "native_capsule_id", *_DIGEST_FIELDS, *_OBSERVATION_FLAGS}
    public = type(value) is dict and value.get("schema") == PUBLIC_SCHEMA
    if public:
        fields.update(("public_validation_contract_sha256", "public_validation_observation_sha256",
            "actual_prior_returncode", "public_validation_status"))
    _need(type(value) is dict and set(value) == fields and value["schema"] in (SCHEMA, PUBLIC_SCHEMA)
        and value["policy_id"] == (PUBLIC_VALIDATION_POLICY if public else DOCTOR_CODING_POLICY)
        and all(_digest(value[key]) for key in _DIGEST_FIELDS)
        and all(type(value[key]) is str and 1 <= len(value[key]) <= 1024
                for key in ("task_cid", "context_cid", "native_capsule_id"))
        and type(value["doctor_advisory_bytes"]) is int and 1 <= value["doctor_advisory_bytes"] <= 32_768
        and type(value["source_files"]) is int and 1 <= value["source_files"] <= 256
        and all(type(value[key]) is type(expected) and value[key] == expected
                for key, expected in _OBSERVATION_FLAGS.items()),
        "invalid Doctor candidate dispatch observation")
    if public:
        _need(all(_digest(value[key]) for key in ("public_validation_contract_sha256", "public_validation_observation_sha256"))
            and type(value["actual_prior_returncode"]) is int and 0 <= value["actual_prior_returncode"] <= 255
            and value["public_validation_status"] in ("passed", "failed", "inconclusive")
            and (value["actual_prior_returncode"] > 0 or value["public_validation_status"] == "failed"),
            "invalid public-validation candidate dispatch observation")
    budget = value["proposal_budget"]
    _need(type(budget) is dict, "closed unchanged Doctor proposal budget required")
    validate_residual_budget(dict(residual_budget=budget,
        token_budget={"residual_" + key: budget.get(key) for key in LIMIT_FIELDS}))
    _need(value["proposal_budget_sha256"] == _sha(canonical_json(budget)),
        "Doctor proposal budget digest differs")
    return json.loads(canonical_json(value))


def validate_doctor_dispatch_projection(*, projection, wire, handoff, descriptor,
                                       repository, workspace):
    """Check immutable constraints plus the complete current Doctor inventory."""
    from . import doctor_residual_context as residual
    from . import doctor_candidate_context as original_context
    from . import local_planning_admission as local
    from .candidate_round_policy import validate_doctor_coding_policy_binding

    from .candidate_round_policy import PUBLIC_VALIDATION_POLICY
    public = (type(projection) is dict and type(projection.get("coding_policy")) is dict
        and projection["coding_policy"].get("policy_id") == PUBLIC_VALIDATION_POLICY)
    _need(type(projection) is dict and set(projection) == {
        "context", "artifact", "advisory_sha256", "advisory_bytes", "coding_policy"} | ({"public_validation"} if public else set()),
        "closed Doctor candidate handoff projection required")
    context, artifact = projection["context"], projection["artifact"]
    admitted = handoff["admission"]
    _need(type(context) is dict and type(artifact) is dict
        and set(artifact) == {"path", "sha256", "bytes"}
        and type(artifact["path"]) is str and type(artifact["sha256"]) is str
        and type(artifact["bytes"]) is int and 0 < artifact["bytes"] <= residual.MAX_BYTES
        and type(wire.get("doctor_advisory")) is str
        and _same(wire.get("doctor_context"), dict(schema="supervisor-doctor-candidate-context-binding@1",
            sha256=_sha(canonical_json(context)), bytes=len(canonical_json(context)))),
        "Doctor candidate prompt differs from protected handoff")
    path = Path(artifact["path"])
    original = context.get("original_doctor")
    _need(type(original) is dict, "captured original Doctor constraints required")
    text, receipt = residual.load_doctor_residual_advisory(artifact=path,
        expected_sha256=artifact["sha256"], repository=repository,
        task_cid=original.get("task_cid"), prompt=wire["native_prompt"],
        require_current_source=True)
    # The loader checks the canonical baseline. The changed allocated worktree
    # is validated separately below against the admitted generation.
    raw = original_context._artifact(path, artifact["sha256"])
    payload = json.loads(raw, object_pairs_hook=residual._unique)
    budget = residual.validate_residual_budget(payload["capsule"])
    baseline = payload["source_manifest"]
    expected_original = dict(schema=original_context.CAPTURE_SCHEMA,
        repository=str(repository), workspace=str(workspace),
        baseline_commit=handoff["baseline_commit"], prior_session_id=descriptor["prior_session_id"],
        descriptor_sha256=descriptor["prior_descriptor_sha256"],
        baseline_binding=handoff["baseline_binding"], native_prompt_sha256=handoff["native_prompt_sha256"],
        task_id=descriptor["task_id"], task_cid=receipt["task_cid"],
        artifact_sha256=artifact["sha256"], artifact_bytes=len(raw),
        context_cid=receipt["context_cid"], native_capsule_id=receipt["native_capsule_id"],
        advisory_sha256=_sha(text.encode()), advisory_bytes=len(text.encode()),
        source_manifest_sha256=_sha(canonical_json(baseline)),
        residual_budget=budget, initial_source_current_verified=True, **original_context.FLAGS)
    _need(_same(original, expected_original) and artifact["bytes"] == len(raw)
        and wire["doctor_advisory"] == text
        and projection["advisory_sha256"] == _sha(text.encode())
        and type(projection["advisory_bytes"]) is int and projection["advisory_bytes"] == len(text.encode()),
        "Doctor candidate obligations, budget or original source binding differ")
    binding = {key: descriptor[key] for key in ("task_id", "native_prompt_sha256", "prior_session_id",
        "workspace_generation", "candidate_scope_cid")}
    binding.update(descriptor_sha256=descriptor["prior_descriptor_sha256"],
        semantic_root_cid=descriptor["candidate_semantic_root_cid"],
        producer_binding_sha256=handoff["producer_binding_sha256"], source_hashes=handoff["source_hashes"])
    selected = set(baseline).intersection(handoff["source_hashes"])
    original_requirements_sha256 = wire["original_requirements_binding"]["original_requirements_sha256"]
    expected_context = dict(schema=original_context.SCHEMA, original_doctor=expected_original,
        candidate_binding=binding, original_requirements_sha256=original_requirements_sha256,
        original_advisory_sha256=_sha(text.encode()), original_advisory_bytes=len(text.encode()),
        candidate_inventory_paths=sorted(selected), unchanged_inventory_paths=sorted(set(baseline)-selected),
        full_doctor_source_inventory_current=True, candidate_evidence_reproved=False,
        original_obligations_preserved=True, residual_budget=budget,
        continuation_status="requires_independent_budget_enforcement", **original_context.FLAGS)
    _need(_same(context, expected_context), "Doctor candidate context generation or coverage differs")
    options = {}
    if public:
        from .candidate_public_validation import validate_public_validation_projection
        validate_public_validation_projection(projection["public_validation"], handoff=handoff,
            wire=wire, descriptor=descriptor, repository=repository, workspace=workspace)
        options = dict(policy_id=PUBLIC_VALIDATION_POLICY,
            public_validation_contract=projection["public_validation"]["contract"])
    policy = validate_doctor_coding_policy_binding(projection["coding_policy"],
        doctor_requirements=expected_original, original_requirements_sha256=original_requirements_sha256,
        deadline_monotonic_ns=admitted["deadline_monotonic_ns"], **options)
    _need(_same(admitted.get("doctor_coding_policy"), policy)
        and admitted.get("doctor_context_sha256") == _sha(canonical_json(context))
        and admitted.get("doctor_advisory_sha256") == projection["advisory_sha256"]
        and type(admitted.get("doctor_advisory_bytes")) is int
        and admitted["doctor_advisory_bytes"] == projection["advisory_bytes"],
        "Doctor coding policy differs from independent owner admission")
    expected_sources = {name: dict(row, sha256=handoff["source_hashes"].get(name, row["sha256"]))
                        for name, row in baseline.items()}
    _need(local._sources(repository, sorted(baseline), max_files=256) == baseline
        and local._sources(workspace, sorted(baseline), max_files=256) == expected_sources
        and original_context._artifact(path, artifact["sha256"]) == raw,
        "complete Doctor source inventory changed before candidate provider dispatch")
    return doctor_dispatch_observation(projection)
