import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import time

from ipfs_datasets_py.logic.formalization.autoencoder import source_function_units as units

output = Path(__file__).resolve().parent
source = output.parent / "source384-construction-lifetime-20261004/trial-01/jobs/supervisor-full-fix-code-vulnerability/fix-code-vulnerability__WCih55g/agent/public-output-evidence/before/bottle.py"
raw = source.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == "761756ce31753e526c48d28ccbca13a5d2493b16fe37aff3e1e4d2efaf3a2bba"
pins = units.pins()
def call():
    return units.extract_function_units(source_bytes=raw, source_sha256=digest,
        source_path="bottle.py", max_functions=1024)
times = []
expected = call()
for iteration in range(5):
    wall, cpu = time.perf_counter(), time.process_time()
    actual = call()
    times.append(dict(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu))
    assert actual == expected
profiler = cProfile.Profile()
profiler.runcall(call)
rows = []
for (path,line,function),(primitive,total,own,cumulative,callers) in pstats.Stats(profiler).stats.items():
    rows.append(dict(path=path,line=line,function=function,primitive_calls=primitive,
        total_calls=total,own_seconds=own,cumulative_seconds=cumulative))
rows.sort(key=lambda r:r["cumulative_seconds"],reverse=True)
assert pins == units.pins() and source.read_bytes() == raw
record = dict(schema="pure-source-unit-extraction-profile@1", source_sha256=digest,
    source_bytes=len(raw),functions=len(expected["units"]),profile=rows[:45],times=times,
    medians={name:statistics.median(row[name] for row in times) for name in times[0]},
    output_sha256=hashlib.sha256(units.wire(expected)).hexdigest(),producer_pins=pins,
    source_and_producer_pins_unchanged=True,solver_calls=0,provider_calls=0,
    source_executed=False,shared_scheduler_access=False,production_qualification=False)
(output/"result.json").write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({key:record[key] for key in ("functions","medians","output_sha256")}))
