#!/usr/bin/env python3
"""Export bounded qualification metadata; never publish source/model/test bodies.

Only explicit local JSON summaries, recorder metadata, and JUnit testcase
attributes are consumed. JUnit failure text, logs, command/environment payloads,
raw provider traces, task instructions, and benchmark verifier/solution files
are never exported. Output must stay beneath the artifact root.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

DEFAULT_FROZEN = '1da7921ed6f436501dda38bacefbecf2e7019a87'
DEFAULT_TREE = '/home/barberb/lift_coding/.worktrees/terminal-expansion-20261005'
FROZEN_DATASETS = '987cf856b2b902aa68c4587bb492b19b932b5d30'
DEFAULT_DATASETS_TREE = '/home/barberb/lift_coding/.worktrees/ir-supervisor-contracts-datasets-20261004'
DEFAULT_FINAL = ('qualification/integration-final-01', 'qualification/deferred-final-01')
SOURCE_HASH_FIELDS = ('owned_source_sha256', 'related_source_sha256', 'target_sha256')
HEX = re.compile(r'[0-9a-f]{64}')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_child(root, name):
    path = root / name
    if path.resolve() != path.absolute() or not path.is_relative_to(root) or '..' in Path(name).parts:
        raise ValueError('artifact path escapes its canonical root')
    return path


def read_json(path):
    if path.resolve(strict=True) != path.absolute() or path.is_symlink():
        raise ValueError('canonical metadata file required')
    raw = path.read_bytes()
    if len(raw) > 4_194_304:
        raise ValueError('bounded metadata required')
    return json.loads(raw)


def evidence(root, path):
    raw = path.read_bytes()
    return dict(artifact=path.relative_to(root).as_posix(), sha256=sha(raw), bytes=len(raw))


def source_map(record, *, before=False):
    nested = record.get('before' if before else 'after', {})
    if 'source_sha256' in nested:
        return dict(nested['source_sha256'])
    merged = {}
    for field in SOURCE_HASH_FIELDS:
        for name, digest in record.get(field, {}).items():
            if name in merged and merged[name] != digest:
                raise ValueError('recorder hash maps disagree')
            merged[name] = digest
    return merged


def target_paths(command):
    result = []
    for value in command.get('argv', []):
        if type(value) is str:
            path = value.split('::', 1)[0]
            if path.endswith('.py') and not path.startswith('-') and Path(path).name.startswith('test_'):
                if len(path.encode()) > 4096 or Path(path).is_absolute() or '..' in Path(path).parts:
                    raise ValueError('noncanonical target path')
                result.append(path)
    if len(set(result)) > 256:
        raise ValueError('selected test-file population exceeds bound')
    return sorted(set(result))


def is_production(name):
    return (name.startswith(('ipfs_accelerate_py/', 'benchmarks/')) and name.endswith('.py')
        and not Path(name).name.startswith(('test_', 'conftest')))


def frozen_digest(tree, revision, path):
    if path.startswith('/') or '..' in Path(path).parts:
        raise ValueError('noncanonical recorded source path')
    result = subprocess.run(['git', '-C', str(tree), 'show', f'{revision}:{path}'], capture_output=True)
    return sha(result.stdout) if result.returncode == 0 else None


def datasets_binding(command, completed, *, revision=FROZEN_DATASETS, tree=None):
    """Replay the recorded datasets tree and changed/selected Python bytes.

    Historical D987 recorders captured only HEAD/clean status. Retain that exact
    narrower claim; newer revisions require explicit tree and source evidence.
    """
    if re.fullmatch(r'[0-9a-f]{40}', revision) is None:
        raise ValueError('full immutable datasets commit required')
    before, after = command.get('before', {}), completed.get('after', {})
    heads = (before.get('datasets_head', command.get('datasets_head')),
             after.get('datasets_head', completed.get('datasets_head')))
    clean = before.get('datasets_status', '') == after.get('datasets_status', '') == ''
    result = dict(datasets_revision_evaluated=revision, datasets_head_before=heads[0],
        datasets_head_after=heads[1], datasets_revision_matches=heads[0] == heads[1] == revision,
        datasets_status_clean_if_recorded=clean, datasets_source_hashes_verified=False,
        datasets_protected_source_count=0, datasets_source_mismatches=[],
        datasets_tree_binding_matches=False, datasets_validation_basis='unqualified',
        datasets_binding_qualified=False)
    recorded = ('datasets_source_sha256' in before or 'datasets_source_sha256' in after
                or 'datasets_tree' in before or 'datasets_tree' in after)
    if not recorded:
        result['datasets_validation_basis'] = 'legacy_recorded_head_and_clean_status'
        result['datasets_binding_qualified'] = (revision == FROZEN_DATASETS
            and result['datasets_revision_matches'] and clean and tree is None)
        return result
    try:
        # The requested tree must equal the recorded selection. Looking up a
        # commit in some other checkout cannot establish which code ran.
        selected = Path(tree if tree is not None else DEFAULT_DATASETS_TREE)
        canonical = selected.resolve(strict=True)
        env = command.get('environment_overrides', {})
        tree_matches = (selected.is_absolute() and canonical == selected
            and before.get('datasets_tree') == after.get('datasets_tree') == str(canonical)
            and env.get('QUALIFICATION_DATASETS') == str(canonical)
            and before.get('datasets_source_base') == after.get('datasets_source_base') == FROZEN_DATASETS)
        result['datasets_tree_binding_matches'] = tree_matches
        if not tree_matches:
            return result
        observed = (before.get('datasets_source_sha256'), after.get('datasets_source_sha256'))
        if any(type(mapping) is not dict or len(mapping) > 65536 for mapping in observed):
            return result
        for mapping in observed:
            if any(type(name) is not str or len(name.encode()) > 4096
                   or Path(name).is_absolute() or '..' in Path(name).parts
                   or not name.endswith('.py') or name != Path(name).as_posix()
                   or (digest is not None and (type(digest) is not str or HEX.fullmatch(digest) is None))
                   for name, digest in mapping.items()):
                return result
        diff = subprocess.run(['git', '-C', str(canonical), 'diff', '--name-only', '-z',
            FROZEN_DATASETS, revision, '--', '*.py'], capture_output=True)
        if diff.returncode or len(diff.stdout) > 4_194_304:
            return result
        required = set(diff.stdout.decode().rstrip('\0').split('\0')) - {''}
        required.update(name for name in target_paths(command)
                        if frozen_digest(canonical, revision, name) is not None)
        protected = sorted(required | set(observed[0]) | set(observed[1]))
        mismatch = [name for name in protected if name not in observed[0] or name not in observed[1]
            or observed[0][name] != observed[1][name]
            or observed[1][name] != frozen_digest(canonical, revision, name)]
        result.update(datasets_validation_basis='recorded_tree_and_changed_or_selected_python_hashes',
            datasets_source_hashes_verified=not mismatch,
            datasets_protected_source_count=len(protected), datasets_source_mismatches=mismatch)
        result['datasets_binding_qualified'] = (result['datasets_revision_matches'] and clean
            and before.get('datasets_status') == after.get('datasets_status') == '' and not mismatch)
    except (OSError, ValueError, UnicodeError):
        # Unavailable explicit evidence is reported as unqualified, not replaced
        # by the legacy fallback or the current checkout's HEAD.
        pass
    return result


def junit(path):
    raw = path.read_bytes()
    if len(raw) > 33_554_432:
        raise ValueError('bounded JUnit metadata required')
    root = ET.fromstring(raw)
    rows = []
    counts = Counter()
    for case in root.findall('.//testcase'):
        identity = (case.get('classname', ''), case.get('name', ''))
        if not all(type(x) is str and len(x.encode()) <= 32768 for x in identity):
            raise ValueError('test identity is missing or unbounded')
        status = ('error' if case.find('error') is not None else
            'failed' if case.find('failure') is not None else
            'skipped' if case.find('skipped') is not None else 'passed')
        counts[status] += 1
        rows.append((identity, status))
    duplicate = len({key for key, _ in rows}) != len(rows)
    # Parameters sometimes contain authored Python snippets: hash the complete
    # identity and export only the unparameterized test-function identifier.
    return rows, dict(counts), duplicate, sha(raw)


def canonical_test_identity(identity, targets):
    """Resolve pytest's root-relative classname against exact selected files.

    Pytest can report ``api.test_x`` or ``test.api.test_x`` for one file when
    collection changes rootdir. Preserve the actual class suffix and complete
    parameterized case name; neither arbitrary prefixes nor display names
    alone establish identity. Ambiguous/unmatched modules cannot be counted.
    """
    classname, name = identity
    parts = classname.split('.')
    if (not name or not parts or len(parts) > 128
            or any(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', part) is None for part in parts)):
        raise ValueError('noncanonical JUnit module/class identity')
    matches = set()
    for target in targets:
        path = Path(target)
        if path.is_absolute() or '..' in path.parts or path.suffix != '.py':
            raise ValueError('canonical selected test file required')
        module = '.'.join(path.with_suffix('').parts)
        for split in range(1, len(parts) + 1):
            prefix = '.'.join(parts[:split])
            if module == prefix or module.endswith('.' + prefix):
                suffix = '.'.join(parts[split:])
                matches.add((target + ('::' + suffix if suffix else ''), name))
    if len(matches) != 1:
        raise ValueError('JUnit identity does not resolve to one selected test file')
    return matches.pop()


def test_run(root, tree, frozen, stem, *, final_candidate,
             datasets_revision=FROZEN_DATASETS, datasets_tree=None):
    base = safe_child(root, stem)
    xml = Path(str(base) + '.xml')
    command_path = Path(str(base) + '-command.json')
    exit_path = Path(str(base) + '-exit.json')
    result = dict(run=stem, source_revision_evaluated=frozen, final_candidate=final_candidate, aggregate_eligible=False)
    if not xml.exists():
        result.update(status='pending', reason='completed JUnit not yet available')
        return result, []
    rows, counts, duplicate, xml_sha = junit(xml)
    result.update(status='completed' if rows else 'zero_completed_cases', counts=counts,
        completed_test_cases=len(rows), junit=evidence(root, xml), duplicate_test_ids=duplicate)
    if not command_path.exists() or not exit_path.exists():
        result.update(reason='no complete before/after recorder; metadata only')
        return result, []
    command, completed = read_json(command_path), read_json(exit_path)
    before, after = source_map(command, before=True), source_map(completed)
    datasets = datasets_binding(command, completed, revision=datasets_revision, tree=datasets_tree)
    targets = target_paths(command)
    try:
        rows = [(canonical_test_identity(identity, targets), status) for identity, status in rows]
    except ValueError:
        result.update(reason='JUnit identity cannot be unambiguously bound to selected test files',
            identity_normalization='unresolved', aggregate_eligible=False)
        return result, []
    duplicate = len({key for key, _ in rows}) != len(rows)
    result.update(identity_normalization='selected-repository-file-and-exact-class-and-case@1',
        duplicate_test_ids=duplicate)
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    protected = sorted({name for name in set(before) | set(after) if is_production(name)} | set(targets))
    mismatch = sorted(name for name in protected if name not in before or before.get(name) != after.get(name)
        or after.get(name) != frozen_digest(tree, frozen, name))
    rc = completed.get('exit_code', completed.get('returncode'))
    result.update(recorder=evidence(root, exit_path), command_metadata=evidence(root, command_path),
        exit_code=rc, seconds=completed.get('seconds', completed.get('duration_seconds')),
        **datasets,
        source_unchanged_reported=completed.get('source_unchanged'),
        source_head_before=command.get('before', {}).get('accelerate_head', command.get('source_head')),
        source_head_after=completed.get('after', {}).get('accelerate_head', completed.get('source_head')),
        production_and_selected_test_sources_match_frozen=bool(protected) and not mismatch,
        protected_source_count=len(protected), selected_test_files=targets,
        protected_source_mismatches=mismatch,
        observed_changed_sources=changed,
        changed_non_target_nonproduction_sources=[name for name in changed if name not in protected],
        xml_digest_matches_recorder=completed.get('xml_sha256') == xml_sha,
        claimed_log_sha256=completed.get('log_sha256'),
        raw_command_environment_or_log_exported=False)
    good = (bool(rows) and type(rc) is int and rc == 0 and not counts.get('failed', 0) and not counts.get('error', 0)
        and not duplicate and bool(targets) and bool(protected) and not mismatch and datasets['datasets_binding_qualified']
        and completed.get('xml_sha256') == xml_sha)
    result['aggregate_eligible'] = bool(final_candidate and good)
    if not rows:
        result['reason'] = 'no completed test cases; collection/preflight failure or empty run excluded'
    elif not final_candidate:
        result['reason'] = 'development or historical run excluded from final totals'
    elif not good:
        result['reason'] = 'final run failed, is stale, or lacks matching source/recorder evidence'
    else:
        result['reason'] = 'final green run with protected source bytes matching frozen revision'
    return result, rows if result['aggregate_eligible'] else []


def profiles(root):
    result = []
    for label in ('tune-mjcf-01', 'constraints-scheduling-01', 'batching-scheduler-03'):
        folder = root / 'task-profiles' / label
        path, neural_path = folder / 'qualification.json', folder / 'source384-qualification.json'
        q, n = read_json(path), read_json(neural_path)
        allowed = ('task_name', 'profile_sha256', 'native_agent_seconds', 'signed_input_count',
            'original_input_sha256', 'indexed_symbols', 'full_capsules', 'disposition', 'seconds',
            'provider_calls', 'input_reconstruction', 'initial_context_qualified',
            'full_supervisor_execution_qualified', 'task_data_semantics_verified', 'official_reward')
        row = {key:q[key] for key in allowed if key in q}
        row['qualification_evidence'] = evidence(root, path)
        neural_allowed = ('status', 'seconds', 'checkpoint_sha256', 'inference_sha256',
            'model_loads', 'source_paths', 'signed_input_paths', 'program_paths', 'task_data',
            'native_worker_executed', 'inference_executed', 'checkpoint_consumed', 'output_created',
            'source_semantics_verified', 'official_reward', 'provider_calls', 'training_steps')
        row['source384'] = {key:n[key] for key in neural_allowed if key in n}
        row['source384']['evidence'] = evidence(root, neural_path)
        # Never export neural candidate IR or summaries containing source text.
        result.append(row)
    return result


def reviews(root, tree, frozen):
    result = []
    for filename in ('independent-review.json', 'independent-data-review.json',
                     'independent-grok-outcome-review.json', 'independent-grok-tools-review.json',
                     'independent-merged-profile-review.json', 'independent-planner180-review.json'):
        path = root / 'qualification' / filename
        if not path.exists():
            continue
        item = read_json(path)
        fields = ('schema', 'reviewer', 'source_head', 'review_method', 'reviewed_scope',
            'findings', 'checks', 'remaining_blocking_findings', 'limitations', 'provider_calls',
            'reviewed_source_unchanged_during_checks', 'model_free_regression_checks',
            'head', 'head_before_merge_commit', 'incoming_commit', 'blocking_findings',
            'resolved_findings', 'review', 'validation')
        exported = {key:item[key] for key in fields if key in item}
        bindings = item.get('source_sha256', {})
        exported['source_sha256'] = bindings
        exported['reviewed_source_matches_frozen'] = bool(bindings) and all(
            frozen_digest(tree, frozen, name) == digest for name, digest in bindings.items())
        exported['evidence'] = evidence(root, path)
        exported['test_counts_are_not_added_to_final_aggregate'] = True
        result.append(exported)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--worktree', type=Path, default=Path(DEFAULT_TREE))
    parser.add_argument('--frozen-source', default=DEFAULT_FROZEN)
    parser.add_argument('--frozen-datasets', default=FROZEN_DATASETS,
        help='Release datasets commit; historical run defaults remain D987 unless --run-datasets is explicit')
    parser.add_argument('--output', required=True, help='New relative directory beneath artifact root')
    parser.add_argument('--final-run', action='append', help='Explicit completed final run stem, repeatable; order defines supersession')
    parser.add_argument('--run-source', action='append', default=[], help='RUN=40-character commit; declares historical per-run qualification honestly')
    parser.add_argument('--run-datasets', action='append', default=[],
        help='RUN=40-character datasets commit; omitted runs keep the historical D987 binding')
    parser.add_argument('--run-datasets-tree', action='append', default=[],
        help='RUN=absolute canonical datasets tree; must match recorded QUALIFICATION_DATASETS')
    args = parser.parse_args(argv)
    root, tree = args.artifact_root.resolve(strict=True), args.worktree.resolve(strict=True)
    if any(re.fullmatch(r'[0-9a-f]{40}', revision) is None
           for revision in (args.frozen_source, args.frozen_datasets)):
        raise ValueError('full immutable source commit required')
    output = safe_child(root, args.output)
    if output.exists():
        raise ValueError('new export directory required; prior evidence is immutable')
    finals = tuple(args.final_run or DEFAULT_FINAL)
    selected = set(finals)
    run_sources = {}
    for entry in args.run_source:
        stem, revision = entry.rsplit('=', 1)
        if stem not in selected or re.fullmatch(r'[0-9a-f]{40}', revision) is None:
            raise ValueError('per-run source requires selected final run and full commit')
        run_sources[stem] = revision
    run_datasets, run_datasets_trees = {}, {}
    for entry in args.run_datasets:
        stem, revision = entry.rsplit('=', 1)
        if (stem not in selected or stem in run_datasets
                or re.fullmatch(r'[0-9a-f]{40}', revision) is None):
            raise ValueError('datasets binding requires one selected run and full commit')
        run_datasets[stem] = revision
    for entry in args.run_datasets_tree:
        stem, value = entry.split('=', 1)
        path = Path(value)
        if (stem not in selected or stem in run_datasets_trees or not path.is_absolute()
                or len(value.encode()) > 4096 or path.resolve(strict=True) != path):
            raise ValueError('datasets tree binding requires one selected run and canonical absolute directory')
        run_datasets_trees[stem] = path
    all_runs = set(finals)
    for directory in ('qualification', 'symbolic', 'task-profiles', 'grok-container'):
        all_runs.update(str(path.relative_to(root))[:-4] for path in (root / directory).glob('*.xml'))
    reports, observations = [], {}
    for stem in [*sorted(all_runs - selected), *finals]:
        run_revision = run_sources.get(stem, args.frozen_source)
        datasets_revision = run_datasets.get(stem, FROZEN_DATASETS)
        report, rows = test_run(root, tree, run_revision, stem, final_candidate=stem in selected,
            datasets_revision=datasets_revision, datasets_tree=run_datasets_trees.get(stem))
        reports.append(report)
        for identity, status in rows:
            observations.setdefault(identity, []).append(dict(status=status, run=stem,
                source_revision=run_revision, datasets_revision=datasets_revision))
    cases = []
    counts = Counter()
    for identity, history in sorted(observations.items()):
        selected_case = history[-1]
        status = selected_case['status']
        counts[status] += 1
        cls, name = identity
        display_name = name.split('[', 1)[0]
        cases.append(dict(case_id_sha256=sha(json.dumps(identity, separators=(',', ':')).encode()),
            class_id_sha256=sha(cls.encode()),
            test_function=display_name if re.fullmatch(r'[A-Za-z0-9_.:-]{1,256}', display_name) else None,
            status=status, selected_final_run=selected_case['run'], source_revision=selected_case['source_revision'],
            datasets_revision=selected_case['datasets_revision'],
            final_green_history=history, superseded_observation_count=len(history)-1))
    qualified = all(next(r for r in reports if r['run'] == stem)['aggregate_eligible'] for stem in finals)
    data_summary = read_json(root / 'task-profiles/qualification-summary.json')['batching_worker_projection']
    summary = dict(schema='terminal-expansion-body-free-qualification@1',
        generated_at_utc=datetime.now(timezone.utc).isoformat(), frozen_source=args.frozen_source,
        frozen_datasets_source=args.frozen_datasets,
        final_qualification_complete=qualified, primary_final_runs=list(finals),
        per_run_source_revisions={stem:run_sources.get(stem,args.frozen_source) for stem in finals},
        per_run_datasets_revisions={stem:run_datasets.get(stem,FROZEN_DATASETS) for stem in finals},
        single_revision_test_claim=len({(run_sources.get(stem,args.frozen_source),
            run_datasets.get(stem,FROZEN_DATASETS)) for stem in finals}) == 1,
        tests=dict(identity_normalization='selected-repository-file-and-exact-class-and-case@1',
            distinct_passed=counts['passed'], distinct_skipped_only=counts['skipped'],
            identity='Unambiguously resolved selected repository test file, exact class suffix and complete parameterized case name; published as SHA256 only.',
            aggregation='Only explicitly selected final green runs with protected hashes matching each declared per-run revision; later listed final runs supersede duplicate case identities. No development, failed, stale, or incomplete runs counted.',
            observed_runs=reports, cases=cases),
        profiles=profiles(root), batching_worker_projection=data_summary,
        independent_reviews=reviews(root, tree, args.frozen_source),
        boundaries=dict(raw_model_or_source_bodies_exported=False,
            credentials_or_command_environment_exported=False,
            benchmark_verifier_or_solution_files_read=False,
            live_benchmark_result_included=False, official_rewards_claimed=False,
            model_provider_tokens_inferred=False, proof_or_completion_authority=False),
        limitations=['Profile checks use exact reviewed public COPY bytes; they are not official container rewards.',
            'Source384 inference and scoped symbolic proofs do not establish whole-program correctness.',
            'Fetch references preserve raw-data obligations and availability, not proof that a model read every byte.',
            'Head-only changes and non-target fixture edits are disclosed separately from production/selected-target byte currentness.',
            'Independent review check counts are metadata and are never added to aggregate test totals.'])
    output.mkdir(parents=True)
    (output / 'qualification-summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + '\n')
    state = 'complete' if qualified else 'pending or blocked'
    readme = f'''# Terminal supervisor expansion qualification
+
+Release source: `{args.frozen_source}`; datasets: `{args.frozen_datasets}`. Declared final local qualification: **{state}**. Both per-run revisions are retained; a mixed-revision aggregate is not a claim that every test ran on the release pair. Historical D987 runs retain their recorded HEAD/clean-status evidence; new datasets revisions additionally require matching recorded tree selection and protected Python source hashes.
+
+The [bounded metadata summary](qualification-summary.json) records **{counts['passed']} distinct passed tests** and **{counts['skipped']} skipped-only cases** from eligible final green runs. Test identities resolve each JUnit module to its unambiguous selected repository file and preserve the exact class suffix and parameterized case name; published identities are hashes to avoid exporting fixture bodies. Development, failed, stale, and unfinished runs remain visible as metadata and contribute no final count. Later explicitly selected final runs supersede duplicate cases from earlier runs.
+
+The three reviewed public profiles exercise exact-input signing, indexed preparation and structured data handling. Tune-MJCF and batching-scheduler separately consumed the pinned Source384 checkpoint with one model load each. The calendar task retains a true empty program index and abstains before checkpoint loading. These local checks do not establish an official Terminal-Bench reward.
+
+The batching scheduler retains 99,183 bytes of immutable JSONL in source blocks and the manifest while using a 32,751-byte worker context. Its 41 full capsules remain in the index; zero fit into this bounded worker projection. Explicit raw fetch references preserve the data obligations without claiming semantic equivalence.
+
+Independent source and data-boundary reviews are included with source-hash currentness. Their separate check counts are not added to the aggregate. No raw instructions, provider responses, credentials, benchmark verifier bodies, or candidate IR bodies are exported.
+
+Live Grok-container trial results must be reported separately from these local qualifications once the controller has inspected the official receipt. No token savings, cross-provider equivalence, whole-program proof, or full-suite completion is claimed here.
+'''.replace('\n+', '\n')
    (output / 'README.md').write_text(readme)
    print(json.dumps(dict(output=str(output), final_qualification_complete=qualified,
        distinct_passed=counts['passed'], distinct_skipped_only=counts['skipped'],
        sha256=sha((output / 'qualification-summary.json').read_bytes()))))


if __name__ == '__main__':
    main()
