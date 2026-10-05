"""Model-free export checks; synthetic metadata fixtures confer no run authority."""
from pathlib import Path
import copy
import ast
import argparse
import hashlib
import importlib.util
import json
import tempfile
import subprocess

SCRIPT_DIR = Path(__file__).resolve().parent

def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

live = module('inspect_live_trials')
qualification = module('export_evidence')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact-root', type=Path,
    help='Optional original artifact root for retained receipt/recorder replay')
parser.add_argument('--worktree', type=Path,
    help='Optional source worktree for producer-enum and protected-source replay')
parser.add_argument('--fixtures-only', action='store_true',
    help='Run portable synthetic checks with sibling modules; no private artifact layout needed')
parser.add_argument('--output-review', type=Path,
    help='Write the bounded check report here; otherwise print it outside the original artifact layout')
args = parser.parse_args()
default_artifacts = SCRIPT_DIR.parent if (SCRIPT_DIR.name == 'qualification'
    and (SCRIPT_DIR.parent / 'task-profiles/qualification-summary.json').is_file()) else None
ROOT = None if args.fixtures_only else (args.artifact_root or default_artifacts)
ROOT = ROOT.resolve(strict=True) if ROOT is not None else None
TREE = None if args.fixtures_only else (args.worktree or (
    Path(qualification.DEFAULT_TREE) if Path(qualification.DEFAULT_TREE).is_dir() else None))
TREE = TREE.resolve(strict=True) if TREE is not None else None
if ROOT is not None and TREE is None:
    parser.error('--worktree is required when replaying original artifacts outside the original workspace')
checks = []

def checked(label, operation):
    operation()
    checks.append(label)

def rejects(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError('invalid metadata accepted')

marker = 'PRIVATE_RAW_BODY_MUST_NOT_BE_EXPORTED'
for projection in [live.usage({'input_tokens': {'body': marker}, 'source': marker, 'unknown': marker}),
    live.usage({'usage': {'total_tokens': marker, 'body': marker}, 'usage_complete_observed': {'body': marker}}, native=True),
    live.invocation({'provider_invocation_policy': {'native_tool_allowlist': [marker],
        'disallowed_tools': [marker], 'native_tool_denylist': [marker]},
        'native_provider_outcome': {'stop_reason': marker, 'reason_code': {'body': marker}}, 'body': marker})]:
    assert marker not in json.dumps(projection)
checks.append('nested bodies and unknown text excluded from usage/outcome/tool fields')
assert live.usage({'total_tokens': 3, 'usage_complete_observed': None})['usage_complete_observed'] is None
assert live.usage({'total_tokens': True})['total_tokens'] is None
assert live.usage({'total_tokens': '3'})['total_tokens'] is None
assert live.usage({'dollar_cost': float('inf')})['dollar_cost'] is None
checks.append('unknown completeness preserved; malformed or nonfinite numbers not promoted')
policy = live.invocation({'provider_invocation_policy': {'native_tool_allowlist': ['read_file'],
    'native_tool_denylist': ['read_file', 'search_tool', 'use_tool'],
    'effective_toolset_verified': False}})
assert policy['provider_invocation_policy']['native_tool_denylist'] == ['read_file', 'search_tool', 'use_tool']
assert policy['requested_policy_is_actual_tool_observation'] is False
assert policy['provider_invocation_policy']['effective_toolset_verified'] is False
checks.append('requested allow/deny lists preserved without actual-tool claim')

provider = dict(attempted=True, status='malformed', reason_code='prose_wrapper',
    request_bytes=22711, response_bytes=153, latency_ms=41233, timeout_ms=180000,
    response=marker, provider_message=marker)
flat = live.planning(dict(qualified=False, planning_strategy='direct',
    provider='grok_cli', model='grok-4.7', provider_calls=1, provider_receipt=provider,
    failure={'type': 'PromptGoalPlannerError', 'message': marker}))
assert flat['provider_receipt']['status'] == 'malformed'
assert flat['provider_receipt']['reason_code'] == 'prose_wrapper'
assert flat['provider_receipt']['response_bytes'] == 153
assert flat['failure_exception_type'] == 'PromptGoalPlannerError'
assert flat['provider_receipt_shape'] == 'flat_provider_receipt'
assert marker not in json.dumps(flat)
checks.append('flat planner failure retains bounded rejection/bytes/type without response or message')
nested = live.planning({'provider_receipt': {'provider': {**provider,
    'status': 'succeeded', 'reason_code': 'provider_graph_accepted'}, 'graph': marker}})
assert nested['provider_receipt']['status'] == 'succeeded'
assert nested['provider_receipt_shape'] == 'planning_receipt.provider'
assert marker not in json.dumps(nested)
checks.append('successful nested planning receipt exports only bounded provider metadata')
symbolic = live.planning(dict(provider='disabled', model='none', provider_calls=0,
    planning_strategy='intent_symbolic', provider_receipt=None))
assert symbolic['provider'] == 'disabled' and symbolic['model'] == 'none'
assert symbolic['provider_calls'] == 0 and symbolic['provider_receipt'] is None
assert symbolic['planning_strategy'] == 'intent_symbolic'
assert live.scalar('provider', 'disabled') == 'unknown'
checks.append('disabled symbolic planning route survives without relaxing recorded Grok selection')
invalid = live.planning(dict(provider=marker, model=marker, planning_strategy=marker,
    provider_receipt=dict(status=marker, reason_code=marker, request_bytes=True,
        response_bytes='153', latency_ms=float('inf'), timeout_ms=-1),
    failure={'type': marker, 'message': marker}))
assert marker not in json.dumps(invalid)
assert all(invalid['provider_receipt'][key] is None for key in
           ('request_bytes', 'response_bytes', 'latency_ms', 'timeout_ms'))
assert invalid['failure_exception_type'] == 'unknown'
checks.append('unknown planner strings and malformed numeric metadata cannot escape closed projection')
if TREE is not None:
    contracts = TREE / 'ipfs_accelerate_py/agent_supervisor/control/control_contracts.py'
    definition = next(node for node in ast.parse(contracts.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == 'OperationStatus')
    operation_statuses = {node.value.value for node in definition.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)}
    assert operation_statuses == live.ENUMS['lifecycle_operation_status']
    assert all(live.scalar('lifecycle_operation_status', status) == status for status in operation_statuses)
    assert live.scalar('lifecycle_operation_status', 'running') == 'unknown'
    checks.append('lifecycle operation status exactly matches producer enum; process state not misclassified')

with tempfile.TemporaryDirectory(prefix='inspector-fixture-') as temporary:
    root = Path(temporary)
    archive = root / 'archive'; archive.mkdir()
    manifest = {'archive_sha256': 'a' * 64}
    raw = json.dumps(manifest).encode(); (archive / 'manifest.json').write_bytes(raw)
    prep = dict(task='tune-mjcf', arm='full', model='grok-4.7', cli_version='1.0.46',
        provider_profile='grok-4.7-cli-1.0.46@1', reasoning_effort='none',
        resource_profile='source384-5cpu-16gib-planner180@1', archive=str(archive),
        archive_sha256='a' * 64, manifest_sha256=hashlib.sha256(raw).hexdigest())
    folder = root / 'grok-tune-mjcf-01'; folder.mkdir()
    (folder / 'preparation.json').write_text(json.dumps(prep))
    selected, bindings = live.selection(root, {}, folder)
    assert selected['resource_profile'] == prep['resource_profile']
    assert bindings['selection_origins']['resource_profile'] == 'preparation'
    assert bindings['source_revisions'] is None and bindings['source_revision_basis'] == 'unknown'
    checks.append('missing receipt selection uses named preparation provenance; absent source revision stays unknown')
    rejects(lambda: live.selection(root, {'resource_profile': 'source384-5cpu-12gib@1'}, folder))
    checks.append('conflicting receipt/preparation selections refused')
    review = dict(qualified=True, manifest_sha256=prep['manifest_sha256'],
        archive_sha256_declared=prep['archive_sha256'],
        source_heads={name: ('b' if label == 'source' else 'c') * 40 for label, name in live.HEAD_PATHS.items()})
    reviews = root / 'grok-container'; reviews.mkdir()
    review_path = reviews / 'archive-review-fixture.json'; review_path.write_text(json.dumps(review))
    selected, bindings = live.selection(root, {}, folder)
    assert bindings['source_revisions']['source'] == 'b' * 40
    assert bindings['source_revision_evidence']['path'] == 'grok-container/archive-review-fixture.json'
    checks.append('source revision comes only from review matching both prepared archive digests')
    review['manifest_sha256'] = 'f' * 64; review_path.write_text(json.dumps(review))
    assert live.selection(root, {}, folder)[1]['source_revisions'] is None
    checks.append('unmatched source review never supplies a revision')
    trial = {'trials': [{'reward': {'reward': 0}}]}
    (folder / 'receipt.json').write_text(json.dumps(trial))
    eigen = root / 'grok-largest-eigenval-01'; eigen.mkdir()
    (eigen / 'preparation.json').write_text(json.dumps({**prep, 'task': 'largest-eigenval'}))
    (eigen / 'receipt.json').write_text(json.dumps(trial))
    rows = live.collect(root)
    assert len(rows) == 2 and len({row['trial_name'] for row in rows}) == 2
    assert {row['task'] for row in rows} == {'tune-mjcf', 'largest-eigenval'}
    checks.append('same attempt number across two task names remains distinct')
    (folder / 'receipt.json').write_text(json.dumps({'trials': [{}, {}]}))
    rejects(lambda: live.collect(root))
    checks.append('multiple native trials cannot be silently reduced to first entry')
    (folder / 'receipt.json').write_text('{"trials":[],"trials":[]}')
    rejects(lambda: live.read_metadata(root, folder / 'receipt.json'))
    checks.append('duplicate metadata keys refused')
    (folder / 'alias.json').symlink_to(folder / 'preparation.json')
    rejects(lambda: live.read_metadata(root, folder / 'alias.json'))
    checks.append('symlink metadata refused')
    empty = root / 'empty.xml'; empty.write_text('<testsuites><testsuite tests="0" errors="0"/></testsuites>')
    assert qualification.junit(empty)[0] == []
    checks.append('empty JUnit parsed as zero observations without aborting exporter')

if ROOT is not None:
    stem = 'qualification/integration-incoming-final-01'
    command = json.loads((ROOT / (stem + '-command.json')).read_text())
    revision = command['before']['accelerate_head']
    for selected in (False, True):
        report, rows = qualification.test_run(ROOT, TREE, revision, stem,
            final_candidate=selected)
        assert not rows and report['aggregate_eligible'] is False and report['status'] == 'zero_completed_cases'
        assert 'no completed test cases' in report['reason']
    checks.append('retained actual collection failure excluded whether historical or selected final')

with tempfile.TemporaryDirectory(prefix='datasets-binding-') as temporary:
    tree = Path(temporary)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tree), '-c', 'user.name=Fixture',
            '-c', 'user.email=fixture@example.invalid', *args], stderr=subprocess.PIPE).decode().strip()
    git('init', '-q', '--initial-branch=main')
    source = tree / 'ipfs_datasets_py/logic/owner.py'; source.parent.mkdir(parents=True)
    source.write_text('VALUE = 1\n')
    target = tree / 'tests/test_binding.py'; target.parent.mkdir(); target.write_text('def test_binding(): pass\n')
    git('add', '.'); git('commit', '-q', '-m', 'base fixture')
    base = git('rev-parse', 'HEAD')
    source.write_text('VALUE = 2\n'); git('add', '.'); git('commit', '-q', '-m', 'changed fixture')
    revision = git('rev-parse', 'HEAD')
    recorded = dict(datasets_head=revision, datasets_tree=str(tree), datasets_source_base=base,
        datasets_status='', datasets_source_sha256={p.relative_to(tree).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, target)})
    command = dict(before=recorded, argv=['tests/test_binding.py'],
        environment_overrides={'QUALIFICATION_DATASETS': str(tree)})
    completed = {'after': copy.deepcopy(recorded)}
    historical_base = qualification.FROZEN_DATASETS
    try:
        qualification.FROZEN_DATASETS = base
        def binding(c=command, e=completed, rev=revision, selected=tree):
            return qualification.datasets_binding(c, e, revision=rev, tree=selected)
        good = binding()
        assert good['datasets_binding_qualified'] and good['datasets_source_hashes_verified']
        assert good['datasets_protected_source_count'] == 2
        checks.append('new datasets binding verifies actual committed production and selected test bytes')
        legacy = {'before': {'datasets_head': base, 'datasets_status': ''}}
        legacy_exit = {'after': copy.deepcopy(legacy['before'])}
        assert binding(legacy, legacy_exit, base, None)['datasets_binding_qualified']
        assert not binding(legacy, legacy_exit, revision, None)['datasets_binding_qualified']
        unbound_new = {'before': {'datasets_head': revision, 'datasets_status': ''}}
        assert not binding(unbound_new, {'after': unbound_new['before']}, revision, None)['datasets_binding_qualified']
        assert not binding(rev=base)['datasets_binding_qualified']
        checks.append('historical datasets receipts cannot be relabeled and new revisions require hashes')
        missing = copy.deepcopy(command)
        del missing['before']['datasets_source_sha256'][source.relative_to(tree).as_posix()]
        missing_exit = {'after': copy.deepcopy(missing['before'])}
        assert not binding(missing, missing_exit)['datasets_binding_qualified']
        omitted_test = copy.deepcopy(command)
        del omitted_test['before']['datasets_source_sha256']['tests/test_binding.py']
        assert not binding(omitted_test, {'after': copy.deepcopy(omitted_test['before'])})['datasets_binding_qualified']
        checks.append('omitted changed production or selected datasets test hash fails closed')
        forged = copy.deepcopy(command)
        forged['before']['datasets_source_sha256']['ipfs_datasets_py/logic/owner.py'] = '0' * 64
        assert not binding(forged, {'after': copy.deepcopy(forged['before'])})['datasets_binding_qualified']
        drifted = copy.deepcopy(completed)
        drifted['after']['datasets_source_sha256']['ipfs_datasets_py/logic/owner.py'] = '1' * 64
        assert not binding(e=drifted)['datasets_binding_qualified']
        dirty = copy.deepcopy(completed); dirty['after']['datasets_status'] = ' M ipfs_datasets_py/logic/owner.py'
        assert not binding(e=dirty)['datasets_binding_qualified']
        checks.append('forged hashes, before-after drift, and dirty datasets state exclude qualification')
        wrong = copy.deepcopy(command); wrong['environment_overrides']['QUALIFICATION_DATASETS'] = str(tree / 'other')
        assert not binding(wrong)['datasets_binding_qualified']
        other = tree / 'other'; other.mkdir()
        assert not binding(selected=other)['datasets_binding_qualified']
        missing_tree = copy.deepcopy(command); del missing_tree['before']['datasets_tree']
        assert not binding(missing_tree)['datasets_binding_qualified']
        checks.append('datasets tree must match requested path plus before-after and actual command selection')
        unsafe = copy.deepcopy(command)
        unsafe['before']['datasets_source_sha256']['../outside.py'] = '0' * 64
        assert not binding(unsafe)['datasets_binding_qualified']
        assert not binding(selected=tree / 'missing')['datasets_binding_qualified']
        checks.append('noncanonical dataset source paths and unavailable explicit trees are refused')
    finally:
        qualification.FROZEN_DATASETS = historical_base

observed = []
if ROOT is not None:
    observed = live.collect(ROOT)
    assert {row['trial_name'] for row in observed} >= {'grok-tune-mjcf-05', 'grok-tune-mjcf-06'}
    for row in observed:
        if row['trial_name'] in {'grok-tune-mjcf-03', 'grok-tune-mjcf-05', 'grok-tune-mjcf-06'}:
            assert row['native_usage']['usage_complete_observed'] is None
    checks.append('completed live metadata projects exact selections and retains unknown native completeness')
    for name, size in [('grok-tune-mjcf-06', 4748), ('grok-tune-mjcf-07', 153)]:
        projected = next(row for row in observed if row['trial_name'] == name)['planning']
        assert projected['planning_strategy'] == 'direct'
        assert projected['provider_receipt']['status'] == 'malformed'
        assert projected['provider_receipt']['reason_code'] == 'prose_wrapper'
        assert projected['provider_receipt']['response_bytes'] == size
        assert projected['failure_exception_type'] == 'PromptGoalPlannerError'
    checks.append('actual attempts 06 and 07 retain prose-wrapper rejection and exact response bytes')
result = dict(schema='body-free-export-regression-check@3', checks=checks, passed=len(checks),
    model_calls=0, live_summary_written=False, trial_observation_count=len(observed),
    source_sha256={name: hashlib.sha256((SCRIPT_DIR / name).read_bytes()).hexdigest()
        for name in ('inspect_live_trials.py', 'export_evidence.py', 'check_evidence_exports.py', 'run.py')
        if (SCRIPT_DIR / name).is_file()},
    artifact_metadata_replay_executed=ROOT is not None, source_enum_replay_executed=TREE is not None,
    synthetic_fixture_results_are_not_benchmark_or_test_aggregate_authority=True)
path = args.output_review or (ROOT / 'qualification/live-export-safety-review.json' if ROOT else None)
if path is not None:
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': len(checks), 'review': str(path), 'live_summary_written': False}))
else:
    print(json.dumps(result, indent=2))
