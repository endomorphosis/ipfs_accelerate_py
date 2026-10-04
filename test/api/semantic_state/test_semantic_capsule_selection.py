"""Native selection parity and immutable bundle corruption boundaries."""
from dataclasses import replace
import json
import re
from types import MappingProxyType

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_selection import (
    immutable_bundle_capsule_reader, select_worker_capsules,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    _scan_scoped_sources, prepare_semantic_context,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.capsules import admit_capsule
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
    IpfsDatasetsSemanticStateProvider,
)


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def original_selection(*, bundle, view, provider, symbols, sources, required,
                       worker_query, worker_capsule_limit, worker_max_bytes):
    """Frozen pre-optimization algorithm, including admission before budget."""
    candidates = list(symbols)
    if worker_query:
        terms = set(re.findall(r"[a-z][a-z0-9_]+", worker_query.lower()))
        candidates.sort(key=lambda symbol: (
            -len(terms.intersection(re.findall(r"[a-z][a-z0-9_]+",
                (symbol.qualified_name + " " + symbol.module_path).lower()))),
            symbol.module_path, symbol.qualified_name, symbol.stable_id,
        ))
    capsules, admissions, selected = [], [], []
    used = 8192 + sum(len(sources[name]) for name in required)
    for symbol in candidates:
        if worker_query and len(capsules) >= worker_capsule_limit:
            break
        capsule = view.capsule(symbol.stable_id)
        freshness = provider.assess_capsule_freshness(capsule, current_state=view)
        admission = admit_capsule(capsule, semantic_state_root_cid=view.root.root_cid,
            assessment=freshness, force_raw_source=symbol.module_path in required)
        payload = capsule.to_dict()
        if worker_query:
            size = len(encoded(payload)) + len(encoded(admission.to_dict())) + 1600
            if used + size > worker_max_bytes:
                continue
            used += size
            selected.append({"path": symbol.module_path,
                             "qualified_name": symbol.qualified_name,
                             "stable_symbol_id": symbol.stable_id})
        capsules.append(payload)
        admissions.append(admission)
    return capsules, admissions, selected


@pytest.fixture(scope="module")
def native_scope():
    sources = {
        "code.py": (
            'def alpha_large(value):\n    """' + 'λ説明' * 1800
            + '"""\n    return value + 1\n\n'
            + "\n".join(f"def zeta_{i}(value):\n    return value + {i}\n" for i in range(6))
        ).encode(),
        "required.py": b"def raw_required(value):\n    return value\n",
        "instruction.md": "Preserve café and λ verbatim.\n".encode(),
    }
    provider = IpfsDatasetsSemanticStateProvider()
    state = _scan_scoped_sources(sources, repository_id="capsule-selection-parity", max_symbols=32)
    bundle = provider.build_semantic_state(state)
    view = provider.view_semantic_state_bundle(bundle)
    return dict(bundle=bundle, view=view, provider=provider, symbols=state.symbols,
                sources=sources, required=("instruction.md", "required.py"))


def comparable(result):
    capsules, admissions, symbols = result
    return encoded([capsules, [item.to_dict() for item in admissions], symbols])


@pytest.mark.parametrize("query,limit,budget", [
    ("alpha_large zeta code", 0, 32768),
    ("alpha_large zeta code", 8, 8192),
    ("alpha_large zeta code", 8, 16384),
    ("alpha_large zeta code", 8, 32768),
    ("alpha_large zeta code", 2, 65536),
    ("raw_required λ", 8, 32768),
    ("", 0, 8192),
])
def test_exact_native_payload_admission_order_and_budget_parity(native_scope, query, limit, budget):
    args = dict(native_scope, worker_query=query, worker_capsule_limit=limit,
                worker_max_bytes=budget)
    assert comparable(select_worker_capsules(**args)) == comparable(original_selection(**args))


def test_oversized_first_candidate_skips_only_its_freshness_and_keeps_later_small(native_scope):
    class CountFreshness:
        def __init__(self):
            self.ids = []

        def assess_capsule_freshness(self, capsule, *, current_state):
            self.ids.append(capsule.stable_symbol_id)
            return native_scope["provider"].assess_capsule_freshness(
                capsule, current_state=current_state)

    old, new = CountFreshness(), CountFreshness()
    args = dict(native_scope, worker_query="alpha_large", worker_capsule_limit=8,
                worker_max_bytes=22000)
    before = original_selection(**dict(args, provider=old))
    after = select_worker_capsules(**dict(args, provider=new))
    assert comparable(before) == comparable(after)
    assert after[0], "fixture must retain at least one later, small capsule"
    oversized = next(s for s in args["symbols"] if s.qualified_name.endswith(".alpha_large"))
    assert oversized.stable_id == old.ids[0]
    assert oversized.stable_id not in new.ids
    assert set(new.ids) < set(old.ids)


def test_fitting_candidate_still_requires_native_freshness(native_scope):
    class RejectFreshness:
        def assess_capsule_freshness(self, capsule, *, current_state):
            raise RuntimeError("freshness unavailable")

    args = dict(native_scope, provider=RejectFreshness(), worker_query="raw_required",
                worker_capsule_limit=8, worker_max_bytes=65536)
    with pytest.raises(RuntimeError, match="freshness unavailable"):
        select_worker_capsules(**args)


def test_exact_admission_byte_boundary_matches_previous_policy(native_scope):
    symbol = next(s for s in native_scope["symbols"] if s.qualified_name.endswith(".raw_required"))
    args = dict(native_scope, symbols=[symbol], worker_query="raw_required",
                worker_capsule_limit=1, worker_max_bytes=65536)
    chosen = original_selection(**args)
    boundary = (8192 + sum(len(args["sources"][name]) for name in args["required"])
                + len(encoded(chosen[0][0])) + len(encoded(chosen[1][0].to_dict())) + 1600)
    for delta, count in ((-1, 0), (0, 1), (1, 1)):
        policy = dict(args, worker_max_bytes=boundary + delta)
        actual = select_worker_capsules(**policy)
        assert len(actual[0]) == count
        assert comparable(actual) == comparable(original_selection(**policy))


def test_index_verified_once_and_each_capsule_matches_native_view(native_scope, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts.semantic_state import models
    original = models.verify_block_bytes
    reads = []

    def count(cid, raw):
        reads.append(cid)
        return original(cid, raw)

    monkeypatch.setattr(models, "verify_block_bytes", count)
    bundle, view = native_scope["bundle"], native_scope["view"]
    reader = immutable_bundle_capsule_reader(bundle, root_cid=view.root.root_cid)
    for symbol in native_scope["symbols"]:
        assert reader(symbol.stable_id).to_dict() == view.capsule(symbol.stable_id).to_dict()
    assert reads.count(bundle.root.capsule_index_cid) == 1


def _unchecked_bundle(bundle, blocks):
    """Corrupt a sealed instance to test the reader's independent byte checks."""
    damaged = object.__new__(type(bundle))
    object.__setattr__(damaged, "root", bundle.root)
    object.__setattr__(damaged, "blocks", MappingProxyType(dict(blocks)))
    return damaged


@pytest.mark.parametrize("target", ["index", "capsule"])
@pytest.mark.parametrize("damage", ["missing", "wrong_bytes", "noncanonical"])
def test_missing_or_tampered_native_blocks_are_rejected(native_scope, target, damage):
    bundle, view = native_scope["bundle"], native_scope["view"]
    symbol = native_scope["symbols"][0]
    cid = bundle.root.capsule_index_cid if target == "index" else view.capsule(symbol.stable_id).capsule_cid
    blocks = dict(bundle.blocks)
    if damage == "missing":
        del blocks[cid]
    elif damage == "wrong_bytes":
        blocks[cid] = b'{}'
    else:
        blocks[cid] += b'\n'
    damaged = _unchecked_bundle(bundle, blocks)
    with pytest.raises((ValueError, KeyError)):
        reader = immutable_bundle_capsule_reader(damaged, root_cid=view.root.root_cid)
        reader(symbol.stable_id)


@pytest.mark.parametrize("damage", ["wrong_symbol", "wrong_kind"])
def test_valid_cid_with_wrong_indexed_identity_is_rejected(native_scope, damage):
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        SemanticStateBundle, SortedPairIndex, canonical_dag_json_bytes,
    )
    bundle, view = native_scope["bundle"], native_scope["view"]
    first, second = native_scope["symbols"][:2]
    replacement = (view.capsule(second.stable_id).capsule_cid if damage == "wrong_symbol"
                   else bundle.root.capsule_index_cid)
    index = SortedPairIndex(pairs=((first.stable_id, replacement),))
    blocks = dict(bundle.blocks)
    blocks[index.index_cid] = canonical_dag_json_bytes(index.identity_payload())
    altered = SemanticStateBundle(root=replace(bundle.root, capsule_index_cid=index.index_cid), blocks=blocks)
    reader = immutable_bundle_capsule_reader(altered, root_cid=altered.root.root_cid)
    with pytest.raises(ValueError):
        reader(first.stable_id)


def test_reader_refuses_foreign_root_mutable_or_injected_bundle(native_scope):
    bundle = native_scope["bundle"]
    with pytest.raises(ValueError, match="immutable native bundle/root"):
        immutable_bundle_capsule_reader(bundle, root_cid=bundle.root.capsule_index_cid)
    damaged = _unchecked_bundle(bundle, bundle.blocks)
    object.__setattr__(damaged, "blocks", dict(bundle.blocks))
    with pytest.raises(ValueError, match="immutable native bundle/root"):
        immutable_bundle_capsule_reader(damaged, root_cid=bundle.root.root_cid)
    class InjectedBundle(type(bundle)):
        def get_block(self, cid):
            raise AssertionError("untrusted reader called")
    injected = InjectedBundle(root=bundle.root, blocks=bundle.blocks)
    with pytest.raises(ValueError, match="immutable native bundle/root"):
        immutable_bundle_capsule_reader(injected, root_cid=bundle.root.root_cid)


def test_new_generation_gets_new_capsule_and_native_freshness(native_scope):
    bundle, provider = native_scope["bundle"], native_scope["provider"]
    sources = dict(native_scope["sources"])
    sources["required.py"] = sources["required.py"].replace(b"return value", b"return value + 7")
    state = _scan_scoped_sources(sources, repository_id="capsule-selection-parity", max_symbols=32)
    new_bundle = provider.build_semantic_state(state)
    new_view = provider.view_semantic_state_bundle(new_bundle)
    old_reader = immutable_bundle_capsule_reader(bundle, root_cid=bundle.root.root_cid)
    new_reader = immutable_bundle_capsule_reader(new_bundle, root_cid=new_view.root.root_cid)
    old_symbol = next(s for s in native_scope["symbols"] if s.qualified_name.endswith(".raw_required"))
    new_symbol = next(s for s in state.symbols if s.qualified_name.endswith(".raw_required"))
    assert old_symbol.stable_id == new_symbol.stable_id
    old, new = old_reader(old_symbol.stable_id), new_reader(new_symbol.stable_id)
    assert old.capsule_cid != new.capsule_cid
    assert new.to_dict() == new_view.capsule(new_symbol.stable_id).to_dict()
    assert provider.assess_capsule_freshness(old, current_state=new_view).to_dict() != provider.assess_capsule_freshness(new, current_state=new_view).to_dict()
    assert old_reader(old_symbol.stable_id).capsule_cid == old.capsule_cid


def test_full_worker_payload_pack_and_roots_match_old_selection(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_capsule_selection
    (tmp_path / "code.py").write_text("def alpha(value):\n    return value + 1\n\ndef zeta(value):\n    return alpha(value)\n")
    (tmp_path / "instruction.md").write_text("Preserve λ and café.\n")
    args = dict(repository=tmp_path, paths=["code.py", "instruction.md"],
                required_raw_paths=["instruction.md"], objective="Inspect alpha", task_id="TASK",
                worker_query="alpha", worker_capsule_limit=8, worker_max_bytes=22000)
    new = prepare_semantic_context(**args, output=tmp_path / "new")
    monkeypatch.setattr(semantic_capsule_selection, "select_worker_capsules", original_selection)
    old = prepare_semantic_context(**args, output=tmp_path / "old")
    assert (tmp_path / "new/worker-context.json").read_bytes() == (tmp_path / "old/worker-context.json").read_bytes()
    for key in ("semantic_root_cid", "scope_cid", "pack_cid", "worker_payload_sha256", "worker_projection", "reconstruction"):
        assert new[key] == old[key]
    assert {p.name: p.read_bytes() for p in (tmp_path / "new/blocks").iterdir()} == {p.name: p.read_bytes() for p in (tmp_path / "old/blocks").iterdir()}
