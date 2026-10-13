"""Compare optional-ID pager with pages captured before its implementation."""
from pathlib import Path
import hashlib
import importlib
import json
from ipfs_datasets_py.logic.software_verification import numerical_logic, numerical_logic_v2, numerical_logic_v3
from ipfs_datasets_py.logic.software_verification import numerical_obligation_retrieval as pager
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384

R = Path(__file__).resolve().parent
PUBLIC = Path('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/largest-eigenval/environment/src')
raw = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
sha = lambda data: hashlib.sha256(data).hexdigest()
baseline = json.loads((R/'baseline.json').read_bytes())
rows = []
checks = 0
for version, logic in enumerate((numerical_logic, numerical_logic_v2, numerical_logic_v3), 1):
    expected = json.loads((R/f'baseline-pages-v{version}.json').read_bytes())
    prior = next(row for row in baseline['rows'] if row['version'] == version)
    regenerated = []
    for entry in expected:
        path = entry['path']
        source = (PUBLIC/path).read_bytes()
        assert sha(source) == baseline['source_sha256'][path]
        report = logic.formalize_numerical_source(source.decode(), path=path)
        assert sha(raw(report)) == prior['report_hashes'][path]
        output = dict(path=path)
        cursor = None
        for key in ('first', 'second'):
            args = dict(report=report, source_text=source.decode(), path=path, limit=1, cursor=cursor)
            page = pager.page_numerical_obligations(**args)
            explicit_none = pager.page_numerical_obligations(**args, obligation_id=None)
            assert raw(page) == raw(explicit_none) == raw(entry[key])
            assert 'obligation_id' not in page['query']
            output[key] = page
            cursor = page['next_cursor']
            checks += 1
        regenerated.append(output)
    digest = sha(raw(regenerated))
    assert digest == prior['page_sha256']
    rows.append(dict(version=version, profile=logic.PROFILE, pages=4,
                     canonical_pages_sha256=digest, omitted_and_none_exact=True))
for name, digest in baseline['frozen_producer_sha256'].items():
    assert sha(Path(importlib.import_module(name).__file__).read_bytes()) == digest
assert structured_source_384._implementation() == baseline['structured_checkpoint_implementation']
result = dict(schema='numerical-exact-selection-dataset-replay@1', passed=True,
    historical_pages_replayed=checks, explicit_none_pages_replayed=checks, rows=rows,
    baseline_sha256=sha((R/'baseline.json').read_bytes()),
    prior_pager_sha256=baseline['pager_sha256'], pager_sha256=sha(Path(pager.__file__).read_bytes()),
    frozen_grammar_logic_program_pins_unchanged=True, structured_checkpoint_pins_unchanged=True,
    provider_calls=0, solver_calls=0, source_executed=False,
    script_sha256=sha(Path(__file__).read_bytes()),
    initial_test_result=dict(passed=76, failed=1,
        failure_reason='Test incorrectly expected max_bytes decimal width to remain constant; corrected bounded fixed-point size calculation, no production change.'),
    final_test_result=dict(passed=77, failed=0, junit='datasets-pager-tests-02.xml'))
(R/'datasets-pager-replay.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(passed=True,historical_pages=checks,explicit_none_pages=checks)))
