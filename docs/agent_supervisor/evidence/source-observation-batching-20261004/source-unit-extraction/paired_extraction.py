"""Same-process source-local geometry reuse versus archived span algorithm."""
import ast
import hashlib
import json
from pathlib import Path
import statistics
import time

from ipfs_datasets_py.logic.formalization.autoencoder import source_function_units as units
from ipfs_datasets_py.logic.formalization.autoencoder.security import security_formalization_evaluation as spans

output = Path(__file__).resolve().parent
source = output.parent / "source384-construction-lifetime-20261004/trial-01/jobs/supervisor-full-fix-code-vulnerability/fix-code-vulnerability__WCih55g/agent/public-output-evidence/before/bottle.py"
raw = source.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == "761756ce31753e526c48d28ccbca13a5d2493b16fe37aff3e1e4d2efaf3a2bba"
before_file = output / "security_formalization_evaluation-before.py"
node = next(node for node in ast.parse(before_file.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == '_function_span')
namespace = {'checkpoint_api': spans.checkpoint_api}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(before_file), 'exec'), namespace)
legacy = namespace['_function_span']
original = units._function_span
pins = units.pins()
mode = 'reused'

def selected_span(raw, node, *, _line_index=None):
    if mode == 'legacy':
        return legacy(raw, node)
    return original(raw, node, _line_index=_line_index)

def call():
    return units.extract_function_units(source_bytes=raw, source_sha256=digest,
        source_path='bottle.py', max_functions=1024)

samples = []
try:
    # A common instrumentation wrapper makes the declaration of producer
    # identity equal across both arms; neither arm grants runtime authority.
    units._function_span = selected_span
    expected = call()
    for iteration in range(10):
        for mode in (('legacy','reused') if iteration % 2 == 0 else ('reused','legacy')):
            wall,cpu = time.perf_counter(),time.process_time()
            actual = call()
            samples.append(dict(iteration=iteration,mode=mode,
                cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall))
            assert actual == expected
finally:
    units._function_span = original
assert source.read_bytes() == raw and units.pins() == pins
actual = call()
assert {key:value for key,value in actual.items() if key != 'producer'} == {
    key:value for key,value in expected.items() if key != 'producer'}
record = dict(schema='source-local-line-index-paired-profile@1',source_sha256=digest,
    producer_pins=pins,original_span_file_sha256=hashlib.sha256(before_file.read_bytes()).hexdigest(),
    source_and_producer_pins_unchanged=True,functions=len(actual['units']),samples=samples,
    medians={mode:{clock:statistics.median(row[clock] for row in samples if row['mode']==mode)
             for clock in ('cpu_seconds','wall_seconds')} for mode in ('legacy','reused')},
    alternating_order=True,complete_instrumented_output_equality_all_samples=True,
    uninstrumented_output_equal_except_instrumentation_producer=True,
    production_output_sha256=hashlib.sha256(units.wire(actual)).hexdigest(),
    source_executed=False,solver_calls=0,provider_calls=0,shared_scheduler_access=False,
    production_qualification=False,
    scope='same-process pure extraction; legacy arm rebuilds each span line table using archived implementation; both arms also build one unused table in the new extraction frame; host activity uncontrolled')
(output/'paired-result.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record['medians']))
