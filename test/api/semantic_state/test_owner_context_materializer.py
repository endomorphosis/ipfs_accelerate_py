"""Current producer materialization with explicit authored launch/tokenizer owners.

The scanner, source admission, authenticated capsule reader, freshness assessor,
offline tokenizer and harness provider boundary execute normally. Launch
authority, model compatibility, durable storage and provider responses are
authored fixtures. These tests grant no remote-tokenizer, model, Lean, semantic
patch or benchmark qualification.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget,
    canonical_context_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.runtime.local_prompt_tokenizer import (
    load_owner_prompt_tokenizer,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    _scan_scoped_sources,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import SupervisorMetaIndex
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessDisposition,
    ModelRoute,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
    IpfsDatasetsSemanticStateProvider,
    SourceBlobStale,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.owner_context_materializer import (
    INTERFACE,
    MaterializerModelBinding,
    OwnerContextMaterializer,
    OwnerMaterializationSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
    SemanticInvalidationObligation,
    SemanticInvalidationPlan,
)
from test.api.semantic_state.test_harness import (
    _accept_request,
    _bootstrap,
    _cid,
    _harness,
    _pack,
    _provider,
    _routing,
    _simple_patch,
)
from test.api.test_local_prompt_tokenizer import make_binding

SOURCE_NAMES = {
    "target_source": "pkg/target.py",
    "surrounding_source": "pkg/support.py",
    "test_source": "pkg/test_target.py",
}
PRODUCER_REVISION = _cid("authored-current-producer-implementation-revision")
MODEL_REVISION = _cid("authored-fixture-model-implementation-revision")
AUTHORITY_REVISION = _cid("authored-fixture-launch-authority-revision-v1")
IMPLEMENTATION_REVISION = _cid("authored-materializer-implementation-revision")
DEPENDENCY_DOCSTRING = 'Preserve café and the literal "quoted".'


@dataclass
class CurrentProducer:
    provider: Any
    sources: dict[str, bytes]
    state: Any
    bundle: Any
    blocks: dict[str, bytes]
    raw: dict[str, bytes]
    source_symbols: dict[str, str]
    source_cids: dict[str, str]
    dependency_symbol: str
    dependency_cid: str
    tokenizer: Any
    artifact: Path
    descriptor: dict[str, Any]

    def get_block(self, cid: str) -> bytes:
        return self.blocks[cid]

    def read_source_blob(self, cid: str) -> bytes:
        return self.raw[cid]


@pytest.fixture
def current(tmp_path: Path) -> CurrentProducer:
    # The private scoped scan uses a fresh temporary snapshot. It cannot expand
    # into the checkout's Git tree or execute imported target code.
    sources = {
        "pkg/target.py": (
            "# Preserve café and literal s0, including trailing whitespace.\n"
            "VALUE = 1\n\n"
            "def target(value):\n"
            "    return value + VALUE\n\n"
            "# Outside the function's span.  \n"
        ).encode(),
        "pkg/support.py": (
            "def helper(value):\n"
            f'    """{DEPENDENCY_DOCSTRING}"""\n'
            "    return value + 1\n"
        ).encode(),
        "pkg/test_target.py": b"def test_target():\n    assert 1 + 1 == 2\n",
    }
    provider = IpfsDatasetsSemanticStateProvider()
    state = _scan_scoped_sources(sources, repository_id="repo:test", max_symbols=32)
    bundle = provider.build_semantic_state(state)
    source_symbols: dict[str, str] = {}
    source_cids: dict[str, str] = {}
    for kind, path in SOURCE_NAMES.items():
        symbol = next(
            item for item in state.symbols
            if item.module_path == path and str(item.kind) == "module"
        )
        source_symbols[kind] = symbol.stable_id
        source_cids[kind] = symbol.source_cid
    helper = next(
        item for item in state.symbols
        if item.module_path == "pkg/support.py" and item.qualified_name.endswith("helper")
    )
    blocks = dict(bundle.blocks)
    view = provider.open_verified_view(bundle.root.root_cid, blocks.__getitem__)
    dependency = view.capsule(helper.stable_id)
    assert str(dependency.confidence) == "exact"
    assessment = provider.assess_capsule_freshness(dependency, current_state=view)
    assert str(assessment.freshness) == "fresh"
    assert str(assessment.admission) in {
        "exact_substitute", "conservative_substitute_with_caveats",
    }
    assets = tmp_path / "authored-offline-tokenizer"
    assets.mkdir()
    artifact, expected_sha256, descriptor = make_binding(
        assets, provider="prod-provider", model="authored-fixture-model"
    )
    tokenizer = load_owner_prompt_tokenizer(
        artifact=artifact,
        expected_sha256=expected_sha256,
        provider="prod-provider",
        model="authored-fixture-model",
    )
    raw = {
        source_cids[kind]: sources[path]
        for kind, path in SOURCE_NAMES.items()
    }
    return CurrentProducer(
        provider, sources, state, bundle, blocks, raw, source_symbols,
        source_cids, helper.stable_id, dependency.capsule_cid,
        tokenizer, artifact, descriptor,
    )


class FixtureAuthority:
    """Explicit authored launch policy; no repository/model-derived admission.

    The owner registers exact independent bindings before dispatch. A monotonic
    epoch detects revocation, including an ABA return to the previous bytes.
    Fault injection can nominate malformed producer inputs to check that the
    adapter independently refuses them; it never fabricates producer facts.
    """

    revision_cid = AUTHORITY_REVISION

    def __init__(self, producer: CurrentProducer) -> None:
        self.producer = producer
        self.expected: dict[str, dict[str, Any]] = {}
        self.epoch = 0
        self.active = True
        self.transform = lambda snapshot: snapshot
        self.admitted: dict[str, int] = {}
        self.admit_calls = 0
        self.verify_calls = 0
        self.current_revision = PRODUCER_REVISION
        self.verify_value: Any = True

    def register(self, request: Any, port: Any, policy: Any, model: Any) -> None:
        # Build the owner's allowlist independently of port._bindings. Changing
        # a request, pack, model or policy later cannot authorize its new body.
        self.expected[request.task_id] = {
            "schema": INTERFACE,
            "materializer_revision_cid": port.revision_cid,
            "producer_revision_cid": PRODUCER_REVISION,
            "implementation_revision_cid": IMPLEMENTATION_REVISION,
            "authority_revision_cid": self.revision_cid,
            "repository_id": request.repository_id,
            "task_id": request.task_id,
            "attempt": request.attempt,
            "lane_id": request.lane_id,
            "attempt_key": request.attempt_key,
            "base_commit": request.base_commit,
            "repo_path": request.repo_path,
            "base_tree": request.base_tree,
            "expected_root": request.expected_root.to_dict(),
            "objective": request.objective,
            "scope": request.scope.to_dict(),
            "constraints": {
                **{name: getattr(request, name) for name in (
                    "toolchain_cid", "dependency_lock_cid", "config_cid",
                    "policy_cid", "interface_cid",
                )},
                "policy": policy.to_dict(),
            },
            "context_pack_cid": cid_for_payload(request.context_pack.to_dict()),
            "context_pack": request.context_pack.to_dict(),
            "provider_model": model.to_dict(),
            "post_apply_semantic_inputs": {
                name: getattr(request, name) for name in (
                    "datasets_state_cid", "datasets_semantic_state_root_cid",
                    "capsule_index_cid", "previous_datasets_state_cid",
                    "previous_datasets_semantic_state_root_cid", "delta_cid",
                    "invalidation_cid", "test_selection_cid",
                )
            } | {
                "changed_symbol_ids": list(request.changed_symbol_ids),
                "obligation_cids": list(request.obligation_cids),
            },
        }

    def _current(self, bindings: Mapping[str, Any], revision: str) -> None:
        if not self.active or revision != self.current_revision:
            raise ValueError("authored launch/source revision is no longer current")
        if dict(bindings) != self.expected.get(bindings.get("task_id")):
            raise ValueError("not an explicitly registered authored launch")

    def admit(self, *, bindings: Mapping[str, Any], producer_revision_cid: str):
        self.admit_calls += 1
        self._current(bindings, producer_revision_cid)
        binding_cid = cid_for_payload(bindings)
        admission_cid = cid_for_payload({
            "fixture": "explicit-launch-owner", "binding_cid": binding_cid,
            "epoch": self.epoch,
        })
        snapshot = OwnerMaterializationSnapshot(
            binding_cid=binding_cid,
            admission_cid=admission_cid,
            semantic_state_root_cid=self.producer.bundle.root.root_cid,
            producer_state_cid=self.producer.state.state_cid,
            semantic_index=self.producer.state,
            read_source_blob=self.producer.read_source_blob,
            get_block=self.producer.get_block,
            source_symbols=self.producer.source_symbols,
            capsule_symbols={
                self.producer.dependency_cid: self.producer.dependency_symbol,
            },
        )
        self.admitted[admission_cid] = self.epoch
        return self.transform(snapshot)

    def verify_snapshot(self, *, snapshot, bindings, producer_revision_cid):
        self.verify_calls += 1
        self._current(bindings, producer_revision_cid)
        if self.admitted.get(snapshot.admission_cid) != self.epoch:
            raise ValueError("authored admission was revoked or replaced")
        return self.verify_value


@dataclass
class DispatchFixture:
    current: CurrentProducer
    authority: FixtureAuthority
    model: MaterializerModelBinding
    port: OwnerContextMaterializer
    provider: Any
    durable: Any
    harness: Any
    root: Any
    request: Any

    def inputs(self, request=None) -> dict[str, Any]:
        request = request or self.request
        return {
            "request": request,
            "pack": request.context_pack,
            "pack_cid": cid_for_payload(request.context_pack.to_dict()),
            "base_tree": request.base_tree,
            "provider_id": self.provider.provider_id,
            "materializer_revision_cid": self.port.revision_cid,
            "route": request.routing_decision.route,
            "mode": self.harness.policy.mode,
        }

    def rendered(self, request=None) -> tuple[str, Any]:
        request = request or self.request
        prompt, decision, selected = self.harness._materialize_model_prompt(
            request, request.context_pack,
            cid_for_payload(request.context_pack.to_dict()),
            request.base_tree, request.routing_decision,
        )
        assert selected is self.provider
        return prompt, decision

    def refused(self, request=None) -> Any:
        outcome = self.harness.run(request or self.request)
        assert outcome.result.disposition in {
            HarnessDisposition.REJECTED.value, HarnessDisposition.UNAVAILABLE.value,
        }
        assert self.provider.calls == []
        assert self.durable.read_root(self.request.repository_id) == self.root
        return outcome


@pytest.fixture
def dispatch(current: CurrentProducer) -> DispatchFixture:
    provider = _provider(patch_text=_simple_patch())
    durable, harness = _harness(providers=(provider,), context_materializer=None)
    root = _bootstrap(harness)
    authority = FixtureAuthority(current)
    model = MaterializerModelBinding(
        provider_id=provider.provider_id,
        route=ModelRoute.SMALL_LOCAL_MODEL.value,
        mode=harness.policy.mode,
        model="authored-fixture-model",
        model_revision_cid=MODEL_REVISION,
        tokenizer=current.tokenizer,
    )
    port = OwnerContextMaterializer(
        provider=current.provider,
        producer_revision_cid=PRODUCER_REVISION,
        implementation_revision_cid=IMPLEMENTATION_REVISION,
        authority=authority,
        policy=harness.policy,
        model_bindings=(model,),
    )
    harness.context_materializer = port
    pack = _pack(
        objective="apply safe patch",
        target_source_cid=current.source_cids["target_source"],
        surrounding_source_cid=current.source_cids["surrounding_source"],
        test_source_cid=current.source_cids["test_source"],
        dependency_capsule_cids=[current.dependency_cid],
        assumptions=["Authored launch and model compatibility; no completion authority."],
        exclusions=["Unrelated source is not nominated for this explicit fixture."],
    )
    request = _accept_request(
        root, routing=_routing(ModelRoute.SMALL_LOCAL_MODEL.value),
        context_pack=pack,
        datasets_state_cid=current.state.state_cid,
        datasets_semantic_state_root_cid=current.bundle.root.root_cid,
    )
    authority.register(request, port, harness.policy, model)
    return DispatchFixture(
        current, authority, model, port, provider, durable, harness, root, request,
    )


def test_native_full_sources_capsule_and_actual_count_reach_provider(dispatch):
    prompt, measured = dispatch.rendered()
    expected = dispatch.current.tokenizer.count(prompt)
    assert measured.inputs.context_tokens == expected > 0
    assert measured.inputs.context_tokens != dispatch.request.context_pack.token_totals["total"]
    packet = json.loads(prompt)
    material = packet["materialized_context"]
    for kind, path in SOURCE_NAMES.items():
        assert material["required_sources"][kind] == {
            "source_cid": dispatch.current.source_cids[kind],
            "text": dispatch.current.sources[path].decode(),
        }
    view = dispatch.current.provider.open_verified_view(
        dispatch.current.bundle.root.root_cid, dispatch.current.get_block,
    )
    capsule = view.capsule(dispatch.current.dependency_symbol)
    assert len(material["dependency_capsules"]) == 1
    transported = material["dependency_capsules"][0]
    assert transported["capsule_cid"] == capsule.capsule_cid
    assert decoded_capsule(transported) == capsule.to_dict()
    assert decoded_capsule(transported)["docstring_hint"] == capsule.docstring_hint
    assert "café" in transported["capsule"]["body"]
    assert '\\"quoted\\"' in transported["capsule"]["body"]
    # The canonical producer retains nested AST evidence deeper than the
    # harness transport ceiling. Its complete exact native body is preserved
    # in an explicitly labeled string, not pruned or summarized away.
    assert structural_depth(capsule.to_dict()) > ContextBudget().max_depth
    assert structural_depth(transported["capsule"]) <= ContextBudget().max_depth
    assert packet["constraints"]["policy"] == dispatch.harness.policy.to_dict()
    assert packet["context_pack"]["assumptions"] == list(dispatch.request.context_pack.assumptions)
    assert packet["context_pack"]["exclusions"] == list(dispatch.request.context_pack.exclusions)
    references = material["references"]
    assert [item["source_kind"] for item in references if item["kind"] == "verified_source"] == list(SOURCE_NAMES)
    admission = next(item for item in references if item["kind"] == "capsule_admission")
    assert admission["assessment"]["root_cid"] == view.root.root_cid
    assert admission["admission"]["assessment_cid"] == admission["assessment"]["assessment_cid"]
    result = dispatch.harness.run(dispatch.request)
    assert len(dispatch.provider.calls) == 1
    assert dispatch.provider.calls[0]["prompt"] == prompt
    assert result.result.disposition == HarnessDisposition.REJECTED.value
    assert "simulation_never_verifies_or_promotes" in result.result.reasons
    assert dispatch.durable.read_root(dispatch.request.repository_id) == dispatch.root


def decoded_capsule(transported):
    body = transported["capsule"]
    assert set(body) == {"encoding", "body"}
    assert body["encoding"] == "canonical-json-utf8@1"
    decoded = json.loads(body["body"])
    assert canonical_context_json_bytes(decoded).decode("utf-8") == body["body"]
    return decoded


def structural_depth(value):
    if isinstance(value, Mapping):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return 0
    return 1 + max((structural_depth(child) for child in children), default=0)


def test_function_nomination_retains_full_blob_instead_of_its_span(dispatch):
    symbol = next(
        item for item in dispatch.current.state.symbols
        if item.module_path == "pkg/target.py" and item.qualified_name.endswith("target")
    )
    source = dispatch.current.provider.read_required_source(
        dispatch.current.state, symbol.stable_id,
        expected_producer_state_cid=dispatch.current.state.state_cid,
        read_source_blob=dispatch.current.read_source_blob,
    )
    assert source.span_bytes != source.source_bytes
    dispatch.authority.transform = lambda snapshot: replace(
        snapshot, source_symbols={**dict(snapshot.source_symbols), "target_source": symbol.stable_id},
    )
    material = dispatch.port.materialize(**dispatch.inputs())
    assert material["required_sources"]["target_source"]["text"].encode() == source.source_bytes
    assert "Outside the function's span.  \n" in material["required_sources"]["target_source"]["text"]
    prompt, _ = dispatch.rendered()
    assert dispatch.port.verify(prompt=prompt, **dispatch.inputs()) is True


@pytest.mark.parametrize("fault", ["missing", "corrupt", "race"])
def test_actual_source_reader_failures_refuse_without_provider(dispatch, fault):
    cid = dispatch.current.source_cids["target_source"]
    raw = dispatch.current.raw[cid]
    if fault == "missing":
        del dispatch.current.raw[cid]
    elif fault == "corrupt":
        dispatch.current.raw[cid] = raw + b"# mismatched bytes\n"
    else:
        calls = 0

        def racing_reader(requested):
            nonlocal calls
            if requested == cid:
                calls += 1
                return raw if calls == 1 else raw + b"# changed between producer reads\n"
            return dispatch.current.raw[requested]

        dispatch.authority.transform = lambda snapshot: replace(snapshot, read_source_blob=racing_reader)
    dispatch.refused()


@pytest.mark.parametrize("fault", ["wrong_state", "wrong_source_symbol", "unknown_source_symbol", "wrong_root"])
def test_producer_state_and_source_membership_fail_closed(dispatch, fault):
    def mutate(snapshot):
        if fault == "wrong_state":
            return replace(snapshot, producer_state_cid=_cid("foreign-producer-state"))
        if fault == "wrong_root":
            return replace(snapshot, semantic_state_root_cid=_cid("unavailable-root"))
        symbols = dict(snapshot.source_symbols)
        symbols["target_source"] = (
            symbols["surrounding_source"] if fault == "wrong_source_symbol"
            else _cid("unknown-source-symbol")
        )
        return replace(snapshot, source_symbols=symbols)

    dispatch.authority.transform = mutate
    dispatch.refused()


@pytest.mark.parametrize("fault", ["missing_nomination", "wrong_symbol", "corrupt_capsule", "corrupt_index", "raw_required"])
def test_authenticated_capsule_and_freshness_failures_refuse(dispatch, fault):
    if fault in {"missing_nomination", "wrong_symbol"}:
        dispatch.authority.transform = lambda snapshot: replace(
            snapshot, capsule_symbols=(
                {} if fault == "missing_nomination" else {
                    dispatch.current.dependency_cid: dispatch.current.source_symbols["target_source"],
                }
            ),
        )
    elif fault in {"corrupt_capsule", "corrupt_index"}:
        cid = (
            dispatch.current.dependency_cid if fault == "corrupt_capsule"
            else dispatch.current.bundle.root.capsule_index_cid
        )
        dispatch.current.blocks[cid] += b" "
    else:
        plan = SemanticInvalidationPlan(
            previous_root_cid=None,
            current_root_cid=dispatch.current.bundle.root.root_cid,
            obligations=(SemanticInvalidationObligation(
                subject_id=dispatch.current.dependency_symbol,
                reason_code="raw_source_requirement",
                remediation_kind="retrieve_raw_source",
                confidence="exact",
            ),),
        )
        dispatch.authority.transform = lambda snapshot: replace(snapshot, invalidation=plan)
    dispatch.refused()


@pytest.mark.parametrize("field", [
    "repository_id", "task_id", "attempt", "lane_id", "attempt_key",
    "base_commit", "repo_path", "base_tree", "expected_root", "objective", "scope",
    "toolchain_cid", "dependency_lock_cid", "config_cid", "policy_cid", "interface_cid",
])
def test_changed_launch_fields_cannot_reuse_authored_admission(dispatch, field):
    original = getattr(dispatch.request, field)
    if field == "attempt":
        changed = original + 1
    elif field == "expected_root":
        changed = replace(original, generation=original.generation + 1)
    elif field == "scope":
        changed = replace(original, effect_paths=("pkg/target.py", "pkg/support.py"))
    elif field.endswith("_cid") or field == "base_tree":
        changed = _cid("different-" + field)
    else:
        changed = "different-" + field
    request = replace(dispatch.request, **{field: changed})
    with pytest.raises(ValueError):
        dispatch.port.materialize(**dispatch.inputs(request))
    dispatch.refused(request)


@pytest.mark.parametrize("fault", ["revoked", "revision", "producer_revision", "truthy_nonbool", "aba_during_read"])
def test_authority_currentness_and_exact_true_recheck_are_required(dispatch, fault):
    if fault == "revoked":
        dispatch.authority.active = False
    elif fault == "revision":
        dispatch.authority.revision_cid = _cid("changed-owner-implementation")
    elif fault == "producer_revision":
        dispatch.authority.current_revision = _cid("different-producer-code-import-revision")
    elif fault == "truthy_nonbool":
        dispatch.authority.verify_value = 1
    else:
        original = dispatch.current.read_source_blob
        changed = False

        def revoke_and_restore(cid):
            nonlocal changed
            raw = original(cid)
            if not changed:
                changed = True
                dispatch.authority.active = False
                dispatch.authority.epoch += 1
                dispatch.authority.active = True
            return raw

        dispatch.authority.transform = lambda snapshot: replace(snapshot, read_source_blob=revoke_and_restore)
    dispatch.refused()


def test_changed_harness_policy_is_not_covered_by_frozen_materializer(dispatch):
    dispatch.harness.policy = replace(
        dispatch.harness.policy, require_full_suite_oracle=True,
    )
    outcome = dispatch.refused()
    assert "model_context_not_admitted" in outcome.result.reasons


@pytest.mark.parametrize("fault", [
    "required_text", "capsule_body", "capsule_encoding", "reference", "objective", "scope",
    "constraint", "pack_assumption", "literal_whitespace",
])
def test_exact_prompt_verification_refuses_tampering(dispatch, fault):
    prompt, _ = dispatch.rendered()
    packet = json.loads(prompt)
    if fault == "required_text":
        packet["materialized_context"]["required_sources"]["target_source"]["text"] += "# forged\n"
    elif fault == "capsule_body":
        capsule = packet["materialized_context"]["dependency_capsules"][0]["capsule"]
        body = json.loads(capsule["body"])
        body["signature"] = {}
        capsule["body"] = canonical_context_json_bytes(body).decode("utf-8")
    elif fault == "capsule_encoding":
        packet["materialized_context"]["dependency_capsules"][0]["capsule"]["encoding"] = "unrecognized@1"
    elif fault == "reference":
        packet["materialized_context"]["references"][0]["admission_cid"] = _cid("forged-admission-row")
    elif fault == "objective":
        packet["objective"] = "different objective"
    elif fault == "scope":
        packet["scope"]["allowed_paths"] = ["unrelated/"]
    elif fault == "constraint":
        packet["constraints"]["interface_cid"] = _cid("foreign-interface")
    elif fault == "pack_assumption":
        packet["context_pack"]["assumptions"] = ["claims promoted to exact"]
    else:
        # A semantic reserialization is still different complete prompt bytes.
        packet = None
    altered = (
        json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if packet is not None else prompt + " "
    )
    assert dispatch.port.verify(prompt=altered, **dispatch.inputs()) is False
    assert dispatch.provider.calls == []


def test_verification_performs_fresh_native_source_reads(dispatch):
    reads = []
    original = dispatch.current.read_source_blob

    def captured(cid):
        reads.append(cid)
        return original(cid)

    dispatch.authority.transform = lambda snapshot: replace(snapshot, read_source_blob=captured)
    prompt, _ = dispatch.rendered()
    before = len(reads)
    assert before >= 12  # Two producer reads, three sources, materialize + verify.
    assert dispatch.port.verify(prompt=prompt, **dispatch.inputs()) is True
    assert len(reads) >= before + 6
    assert dispatch.authority.admit_calls >= 3
    assert dispatch.authority.verify_calls >= 6


def test_mutation_after_rendering_is_not_hidden_by_cached_material(dispatch):
    prompt, _ = dispatch.rendered()
    cid = dispatch.current.source_cids["test_source"]
    dispatch.current.raw[cid] += b"# changed after rendering\n"
    with pytest.raises(SourceBlobStale):
        dispatch.port.verify(prompt=prompt, **dispatch.inputs())
    assert dispatch.provider.calls == []


@pytest.mark.parametrize("fault", ["descriptor", "asset"])
def test_actual_local_tokenizer_drift_refuses_dispatch(dispatch, fault):
    path = dispatch.current.artifact
    if fault == "asset":
        path = path.parent / dispatch.current.descriptor["asset_path"]
    path.write_bytes(path.read_bytes() + b" ")
    dispatch.refused()


@pytest.mark.parametrize("fault", ["absent", "wrong_capsule", "wrong_root", "wrong_state"])
def test_injected_assessment_errors_cannot_become_admission(dispatch, monkeypatch, fault):
    # Fault injection wraps an actual current producer assessment; an authored
    # malformed answer must not trigger the compatibility default of "fresh".
    actual = dispatch.current.provider.assess_capsule_freshness

    def faulty(self, capsule, **bindings):
        assessment = actual(capsule, **bindings)
        if fault == "absent":
            return None
        field = {
            "wrong_capsule": "capsule_cid", "wrong_root": "root_cid",
            "wrong_state": "producer_repository_state_cid",
        }[fault]
        return replace(assessment, **{field: _cid("foreign-" + field)})

    # The native provider uses slots. Patch its class in this isolated pytest
    # process; monkeypatch restores the method after each parametrized case.
    monkeypatch.setattr(type(dispatch.current.provider), "assess_capsule_freshness", faulty)
    dispatch.refused()


def test_actual_conservative_producer_capsule_keeps_its_caveats(dispatch):
    current = dispatch.current
    current.state = replace(current.state, symbols=tuple(
        replace(symbol, confidence="conservative")
        if symbol.stable_id == current.dependency_symbol else symbol
        for symbol in current.state.symbols
    ))
    current.bundle = current.provider.build_semantic_state(current.state)
    current.blocks = dict(current.bundle.blocks)
    view = current.provider.open_verified_view(current.bundle.root.root_cid, current.get_block)
    capsule = view.capsule(current.dependency_symbol)
    assert str(capsule.confidence) == "conservative"
    current.dependency_cid = capsule.capsule_cid
    pack = replace(dispatch.request.context_pack, dependency_capsule_cids=(current.dependency_cid,))
    dispatch.request = replace(dispatch.request, context_pack=pack)
    dispatch.authority.epoch += 1
    dispatch.authority.register(dispatch.request, dispatch.port, dispatch.harness.policy, dispatch.model)
    prompt, _ = dispatch.rendered()
    material = json.loads(prompt)["materialized_context"]
    reference = next(item for item in material["references"] if item["kind"] == "capsule_admission")
    assert reference["assessment"]["freshness"] == "fresh"
    assert reference["assessment"]["admission"] == "conservative_substitute_with_caveats"
    assert reference["admission"]["admission"] == "conservative_substitute_with_caveats"
    assert "confidence:conservative" in reference["admission"]["caveats"]
    assert decoded_capsule(material["dependency_capsules"][0]) == capsule.to_dict()
    assert decoded_capsule(material["dependency_capsules"][0])["confidence"] == "conservative"
    dispatch.harness.run(dispatch.request)
    assert dispatch.provider.calls[0]["prompt"] == prompt


def test_complete_encoded_capsule_over_existing_text_ceiling_refuses(dispatch):
    current = dispatch.current
    # Small exact source can produce a much larger full native AST capsule.
    # This is a real producer case, not a fabricated oversized capsule body.
    current.sources["pkg/support.py"] = (
        "def helper(value):\n"
        + "".join(f"    value = value + {number}\n" for number in range(60))
        + "    return value\n"
    ).encode()
    assert len(current.sources["pkg/support.py"]) < ContextBudget().max_text_bytes
    current.state = _scan_scoped_sources(
        current.sources, repository_id="repo:test", max_symbols=32,
    )
    current.bundle = current.provider.build_semantic_state(current.state)
    current.blocks = dict(current.bundle.blocks)
    for kind, path in SOURCE_NAMES.items():
        module = next(
            symbol for symbol in current.state.symbols
            if symbol.module_path == path and str(symbol.kind) == "module"
        )
        current.source_symbols[kind] = module.stable_id
        current.source_cids[kind] = module.source_cid
        current.raw[module.source_cid] = current.sources[path]
    view = current.provider.open_verified_view(current.bundle.root.root_cid, current.get_block)
    capsule = view.capsule(current.dependency_symbol)
    body = canonical_context_json_bytes(capsule.to_dict())
    assert len(body) > ContextBudget().max_text_bytes
    current.dependency_cid = capsule.capsule_cid
    pack = replace(
        dispatch.request.context_pack,
        surrounding_source_cid=current.source_cids["surrounding_source"],
        dependency_capsule_cids=(current.dependency_cid,),
    )
    dispatch.request = replace(dispatch.request, context_pack=pack)
    dispatch.authority.epoch += 1
    dispatch.authority.register(dispatch.request, dispatch.port, dispatch.harness.policy, dispatch.model)
    with pytest.raises(ValueError, match="capsule body exceeds"):
        dispatch.port.materialize(**dispatch.inputs())
    dispatch.refused()


def test_explicit_actual_git_tree_identifier_is_preserved(dispatch, tmp_path):
    repository = tmp_path / "authored-git-source-tree"
    repository.mkdir()
    for name, raw in dispatch.current.sources.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for args in (("init", "-q"), ("add", ".")):
        subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True)
    tree = subprocess.run(
        ["git", "write-tree"], cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert len(tree) in {40, 64} and all(char in "0123456789abcdef" for char in tree)
    dispatch.request = replace(dispatch.request, base_tree=tree)
    dispatch.authority.register(dispatch.request, dispatch.port, dispatch.harness.policy, dispatch.model)
    prompt, _ = dispatch.rendered()
    assert json.loads(prompt)["base_tree"] == tree
    dispatch.harness.run(dispatch.request)
    assert len(dispatch.provider.calls) == 1
    assert dispatch.provider.calls[0]["prompt"] == prompt


def test_adapter_construction_reads_and_count_do_not_activate_metadata_owner(dispatch, monkeypatch):
    prompt, _ = dispatch.rendered()
    activations = []

    def forbidden(*args, **kwargs):
        activations.append((args, kwargs))
        raise AssertionError("materializer must never discover/open a metadata owner")

    monkeypatch.setattr(SupervisorMetaIndex, "from_env", forbidden)
    model = replace(dispatch.model)
    port = OwnerContextMaterializer(
        provider=dispatch.current.provider,
        producer_revision_cid=PRODUCER_REVISION,
        implementation_revision_cid=IMPLEMENTATION_REVISION,
        authority=dispatch.authority,
        policy=dispatch.harness.policy,
        model_bindings=(model,),
    )
    assert port.revision_cid == dispatch.port.revision_cid
    material = port.materialize(**dispatch.inputs())
    assert material["required_sources"]["target_source"]["text"].encode() == dispatch.current.sources["pkg/target.py"]
    assert port.verify(prompt=prompt, **dispatch.inputs()) is True
    assert port.count_tokens(
        prompt, provider_id=dispatch.provider.provider_id,
        route=dispatch.request.routing_decision.route, mode=dispatch.harness.policy.mode,
    ) > 0
    # Mirrors swallow failures; checking the recorded activation list is needed
    # to distinguish suppression from a swallowed attempt to open an owner.
    assert activations == []


@pytest.mark.parametrize("field", ["provider_id", "route", "mode"])
def test_absent_exact_dispatch_mapping_cannot_guess_tokenizer(dispatch, field):
    inputs = dispatch.inputs()
    inputs[field] = "unmapped-" + field
    with pytest.raises(ValueError, match="mapping"):
        dispatch.port.materialize(**inputs)
    assert dispatch.provider.calls == []


def test_model_binding_requires_matching_actual_owner_tokenizer(dispatch):
    with pytest.raises(ValueError, match="mismatch"):
        replace(dispatch.model, model="another-model")
    with pytest.raises(ValueError, match="mismatch"):
        replace(dispatch.model, provider_id="another-provider")


@pytest.mark.parametrize("field", ["producer", "implementation", "authority", "model", "policy"])
def test_declared_authority_model_or_implementation_change_has_new_revision(dispatch, field):
    authority = FixtureAuthority(dispatch.current)
    models = (dispatch.model,)
    policy = dispatch.harness.policy
    producer_revision = PRODUCER_REVISION
    implementation_revision = IMPLEMENTATION_REVISION
    if field == "producer":
        producer_revision = _cid("new-producer-code")
    elif field == "implementation":
        implementation_revision = _cid("new-materializer-code")
    elif field == "authority":
        authority.revision_cid = _cid("new-launch-authority-code")
    elif field == "model":
        models = (replace(dispatch.model, model_revision_cid=_cid("new-model-code")),)
    else:
        policy = replace(policy, require_full_suite_oracle=True)
    port = OwnerContextMaterializer(
        provider=dispatch.current.provider,
        producer_revision_cid=producer_revision,
        implementation_revision_cid=implementation_revision,
        authority=authority,
        policy=policy,
        model_bindings=models,
    )
    assert port.revision_cid != dispatch.port.revision_cid
    with pytest.raises(ValueError, match="revision"):
        port.materialize(**dispatch.inputs())


@pytest.mark.parametrize("overflow", [False, True])
def test_exact_actual_token_ceiling_and_one_token_overflow(dispatch, overflow):
    prompt, measured = dispatch.rendered()
    count = dispatch.current.tokenizer.count(prompt)
    assert count == measured.inputs.context_tokens
    budget = ContextBudget()
    original_window = dispatch.provider.capability_spec.max_context_tokens
    window = count + budget.reserved_output_tokens + budget.reserved_tool_tokens - int(overflow)
    assert window < original_window
    object.__setattr__(dispatch.provider, "spec", replace(
        dispatch.provider.capability_spec, max_context_tokens=window,
    ))
    if overflow:
        outcome = dispatch.refused()
        assert "model_context_overflow" in outcome.result.reasons
    else:
        dispatch.harness.run(dispatch.request)
        assert len(dispatch.provider.calls) == 1
        assert dispatch.provider.calls[0]["prompt"] == prompt


def test_parallel_verifications_do_not_exchange_prompt_admission(dispatch):
    pack = replace(dispatch.request.context_pack, objective="inspect the helper in parallel")
    other = replace(
        dispatch.request, task_id="parallel-independent-task", attempt_key="parallel-independent-attempt",
        objective=pack.objective, context_pack=pack,
    )
    dispatch.authority.register(other, dispatch.port, dispatch.harness.policy, dispatch.model)
    first_prompt, _ = dispatch.rendered()
    second_prompt, _ = dispatch.rendered(other)
    assert first_prompt != second_prompt

    def verify(pair):
        request, prompt, expected = pair
        return dispatch.port.verify(prompt=prompt, **dispatch.inputs(request)) is expected

    pairs = [
        (dispatch.request, first_prompt, True),
        (other, second_prompt, True),
        (dispatch.request, second_prompt, False),
        (other, first_prompt, False),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(verify, pairs))
    assert dispatch.provider.calls == []


def test_current_canonical_producer_imports_are_used():
    import ipfs_datasets_py.logic.software_contracts.semantic_state.api as native_api
    import ipfs_datasets_py.logic.software_contracts.semantic_state.freshness as native_freshness
    import ipfs_datasets_py.logic.software_contracts.semantic_state.source as native_source

    root = Path("/home/barberb/lift_coding/external/ipfs_datasets").resolve()
    for module in (native_api, native_source, native_freshness):
        assert Path(module.__file__).resolve().is_relative_to(root)
