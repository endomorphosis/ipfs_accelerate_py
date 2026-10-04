"""Independent, read-only audit of one retained advisory/native-worker join.

Usage: /usr/bin/python3 -B audit-joined-worker.py OUTPUT
       --manifest MANIFEST --manifest-sha256 EXPECTED [--pin-count N --pin-bytes N]
No project imports, owner constructors, extension LOAD, training, proof or worker
execution. Read-only Git object inspection is the sole subprocess operation.
This verifies historical byte/public-signature correspondence, not process-origin
attestation, current grants, convergence, or learned-feature execution authority.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

sys.dont_write_bytecode = True
import duckdb
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

VERSION = 'independent-advisory-spawn-artifact-audit@1'
PATHS = {'calc.py','decoy.py','support.py','consumer.py','unsupported.py',
         'check_type.py','check_offset.py','known_variant.py','tune.py','canary.py'}
BEFORE = b'def increment(n: int) -> int:\n    return n + 1\n'
AFTER = b'def increment(n: int) -> int:\n    return n + 2\n'
CLAIMS = ('source_semantics_verified','runtime_behavior_verified','proof_authority',
    'execution_authority','publication_authority','completion_authority','omission_authority',
    'production_activated','convergence_proved','generalization_verified','parser_correctness_proved',
    'universal_python_semantics_proved','formal_decoder_available','model_influenced_worker_edit')
FINITE_FALSE = ('source_semantics_verified','runtime_behavior_verified','proof_authority',
    'code_proof_authority','production_admitted','production_activation','execution_authority',
    'completion_authority','mutation_authority','omission_authority','worker_launched','convergence_proved')
CANDIDATE_FALSE = ('source_semantics_verified','runtime_behavior_verified','proof_authority',
    'execution_authority','completion_authority','mutation_authority','publication_authority',
    'production_activation','native_worker_loop_qualified','untrusted_worker_isolated',
    'filesystem_isolation','network_isolation','owner_keys_inaccessible','process_origin_attested','convergence_proved')
LOWERING_FALSE = ('source_semantics_verified','runtime_behavior_verified','behavior_authority',
    'proof_authority','execution_authority','completion_authority','mutation_authority',
    'whole_program_semantics_verified','cpython_equivalence_proved','source_origin_proved',
    'training_convergence_proved','source_parser_correctness_proved','universal_runtime_behavior_proved')
SIGNATURES = []
DATABASES = []
ROOT = NATIVE = ADVISORY = None
NATIVE_METADATA_LIMITS = {'families':32,'rows':65536,'row_bytes':262144,
    'total_payload_bytes':64*1024**2,'source_bytes':1048576,'manifest_bytes':16*1024**2,
    'json_depth':32,'json_nodes':50000,'restart_seconds':60,'history_bytes':256*1024**2,
    'packet_rows':100,'packet_bytes':128*1024,'single_row_packet_bytes':262144+1024}
SHARDED_METADATA_LIMITS = {'part_payload_bytes':32*1024**2,'part_rows':32768,'parts':16,
    'total_payload_bytes':512*1024**2,'rows':524288,'families':32,
    'manifest_bytes':16*1024**2,'export_bytes':256*1024**2,'restart_seconds':300}


def require(value, message):
    if not value:
        raise AssertionError(message)


def wire(value, ascii=False):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=ascii,
                      allow_nan=False).encode('utf-8')


def same(left, right, message):
    require(wire(left) == wire(right), message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return 'sha256:' + sha(wire(value))


def raw_cid(raw):
    return 'b' + base64.b32encode(b'\x01\x55\x12\x20' + hashlib.sha256(raw).digest()).decode().lower().rstrip('=')


def strict_plain(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is list:
        for child in value:
            strict_plain(child)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for child in value.values():
            strict_plain(child)
        return
    raise AssertionError('structured CID must use the exact finite integer/string JSON profile')


def cid(value):
    strict_plain(value)
    return 'b' + base64.b32encode(b'\x01\xa9\x02\x12\x20' + hashlib.sha256(wire(value)).digest()).decode().lower().rstrip('=')


def self_cid(value, key):
    require(value[key] == cid({k:v for k,v in value.items() if k != key}), 'complete self identity: '+key)


def no_duplicates(pairs):
    value = {}
    for key, child in pairs:
        require(key not in value, 'duplicate JSON object key')
        value[key] = child
    return value


def parse(raw):
    return json.loads(raw, object_pairs_hook=no_duplicates,
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError('nonfinite JSON '+token)))


def inside(path, root=None):
    path = Path(path)
    root = ROOT if root is None else Path(root)
    require(path.is_absolute() and path.is_relative_to(root) and path.resolve(strict=True) == path,
            'noncanonical or escaped retained path: '+str(path))
    return path


def mapped(path):
    path = Path(path)
    require('..' not in path.parts, 'parent traversal in artifact path')
    if not path.is_absolute():
        return inside(ROOT/path)
    if path.is_relative_to(ROOT):
        return inside(path)
    if path == Path('/opt/ipfs-supervisor/container-boundary.json'):
        return inside(ROOT/'deployment/container-boundary.json')
    for prefix, destination in (
        ('/results', ROOT), ('/opt/ipfs-supervisor/finite-handoffs', ROOT/'handoffs'),
        ('/opt/ipfs-supervisor/source', ROOT/'source'), ('/opt/ipfs-supervisor/datasets', ROOT/'datasets'),
        ('/opt/ipfs-supervisor/kit', ROOT/'kit'), ('/opt/ipfs-supervisor/bin', ROOT/'deployment')):
        base = Path(prefix)
        if path.is_relative_to(base):
            return inside(destination/path.relative_to(base))
    raise AssertionError('unmapped external artifact path: '+str(path))


def read(path, maximum=512*1024**2, external=False):
    path = Path(path)
    if not external:
        inside(path)
    else:
        require(path.is_absolute() and path.resolve(strict=True) == path, 'external manifest must be canonical')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= maximum, 'bounded regular file required')
        raw = bytearray()
        while len(raw) <= maximum:
            block = os.read(fd, min(65536, maximum+1-len(raw)))
            if not block:
                break
            raw.extend(block)
        after = os.fstat(fd)
        current = path.lstat()
        identity = lambda s: (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        require(len(raw) <= maximum and len(raw) == after.st_size
                and identity(before) == identity(after) == identity(current), 'file drift during independent read')
        return bytes(raw)
    finally:
        os.close(fd)


def load(path):
    return parse(read(path))


def pin(item, base=None):
    path = item['path'] if 'path' in item else item['relative_path']
    if not Path(path).is_absolute() and base is not None:
        path = inside(Path(base)/path)
    else:
        path = mapped(path)
    raw = read(path)
    size = item.get('bytes',item.get('size_bytes'))
    require(type(size) is int and size == len(raw) and type(item['sha256']) is str
            and sha(raw) == item['sha256'].removeprefix('sha256:'), 'exact artifact pin: '+str(path))
    return raw


def false(value, names):
    for name in names:
        require(value.get(name) is False, 'exact false scope/authority flag: '+name)


def public_signature(envelope, binding=None):
    require(type(envelope) is dict and set(envelope) == {'payload','binding'}, 'closed signed envelope')
    actual = envelope['binding']
    require(type(actual) is dict and set(actual) == {'identity','profile_id','signature'}, 'closed public signature binding')
    if binding is not None:
        same({key:actual[key] for key in ('identity','profile_id')}, binding, 'same signed profile identity')
    did = actual['identity']
    require(type(did) is str and did.startswith('did:key:z'), 'Ed25519 public did:key')
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    text = did[9:]
    number = 0
    for character in text:
        require(character in alphabet, 'base58 public identity')
        number = number*58 + alphabet.index(character)
    decoded = b'\0'*(len(text)-len(text.lstrip('1'))) + number.to_bytes((number.bit_length()+7)//8,'big')
    require(len(decoded) == 34 and decoded[:2] == b'\xed\x01', 'exact public Ed25519 multicodec')
    Ed25519PublicKey.from_public_bytes(decoded[2:]).verify(base64.b64decode(actual['signature'],validate=True),
                                                        wire(envelope['payload'],ascii=True))
    SIGNATURES.append({'schema':envelope['payload'].get('schema'), 'envelope_cid':cid(envelope),
                       'identity':did,'profile_id':actual['profile_id']})
    return envelope['payload']


def database(path, parquet=False):
    path = inside(path)
    DATABASES.append(str(path))
    return duckdb.connect(str(path),read_only=True,config={
        'threads':'1','enable_external_access':'true' if parquet else 'false',
        'autoload_known_extensions':'false','autoinstall_known_extensions':'false'})


def rows(connection, table):
    require(re.fullmatch(r'[a-z_]+(?:\.[a-z_]+)?',table) is not None, 'closed SQL table identifier')
    cursor = connection.execute('SELECT * FROM '+table)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names,row)) for row in cursor.fetchall()]


def ordered(values):
    return sorted(values,key=wire)


def successful_process(value):
    require(type(value['returncode']) is int and value['returncode'] == 0
            and value['termination_reason'] == 'completed' and value['workspace_cleaned'] is True
            and value['stderr'] == value['error'] == '', 'historical successful bounded process receipt')
    false(value,('timed_out','cancelled','unavailable','output_truncated','workspace_limit_exceeded',
                 'process_tree_terminated','resource_exhausted'))
    require(type(value['elapsed_ms']) is int and value['elapsed_ms'] >= 0, 'exact process duration')


def owned_artifacts(value):
    output = mapped(value['output'])
    observed = {}
    for role, item in value['artifacts'].items():
        path = mapped(item['path'])
        inside(path,output)
        raw = pin(item)
        require(raw_cid(raw) == item['cid'], 'raw artifact CID: '+role)
        observed[role] = raw
    self_cid(value,'result_cid')
    same(load(output/'result.json'),value,'whole producer-owned result')
    return observed


def reconstruct(bounded):
    chunks = defaultdict(dict)
    consumed = set()
    for chunk in bounded.get('metadata_chunks',[]):
        require(set(chunk) == {'schema','artifact_id','ordinal','sha256','bytes','base64','chunk_id'}
                and chunk['schema'] == 'terminal-codebase-metadata-chunk@1'
                and type(chunk['ordinal']) is int and chunk['ordinal'] >= 0
                and type(chunk['bytes']) is int and 1 <= chunk['bytes'] <= 49152, 'exact bounded metadata chunk')
        require(chunk['chunk_id'] == digest({k:v for k,v in chunk.items() if k != 'chunk_id'}), 'chunk identity')
        raw = base64.b64decode(chunk['base64'],validate=True)
        require(len(raw) == chunk['bytes'] and sha(raw) == chunk['sha256'], 'whole chunk bytes')
        require(chunk['ordinal'] not in chunks[chunk['artifact_id']], 'duplicate chunk ordinal')
        chunks[chunk['artifact_id']][chunk['ordinal']] = raw
    result = {}
    for family, values in bounded.items():
        if family == 'metadata_chunks':
            continue
        result[family] = []
        for ordinal, value in enumerate(values):
            if value.get('schema') != 'terminal-codebase-metadata-artifact@1':
                result[family].append(value)
                continue
            require(set(value) == {'schema','family','original_ordinal','payload_sha256','payload_bytes',
                'chunk_count','encoding','artifact_id'} and value['family'] == family
                and type(value['original_ordinal']) is int and value['original_ordinal'] == ordinal
                and type(value['payload_bytes']) is int and 1 <= value['payload_bytes'] <= 32000000
                and type(value['chunk_count']) is int
                and value['chunk_count'] == (value['payload_bytes']+49152-1)//49152
                and value['encoding'] == 'canonical-json-utf8/base64-chunks'
                and value['artifact_id'] == digest({k:v for k,v in value.items() if k != 'artifact_id'}), 'exact complete artifact descriptor')
            key = value['artifact_id']
            require(key not in consumed and key in chunks and type(value['chunk_count']) is int
                    and set(chunks[key]) == set(range(value['chunk_count'])), 'complete unique chunk population')
            raw = b''.join(chunks[key][i] for i in range(value['chunk_count']))
            require(type(value['payload_bytes']) is int and len(raw) == value['payload_bytes']
                    and len(raw) <= 32000000 and sha(raw) == value['payload_sha256'], 'lossless original producer bytes')
            record = parse(raw)
            require(type(record) is dict and wire(record) == raw, 'canonical complete reconstructed producer object')
            result[family].append(record)
            consumed.add(key)
    require(consumed == set(chunks),'orphan or unused metadata chunks')
    return result


def native_metadata_audit(directory):
    manifest = load(directory/'manifest.json')
    require(manifest['schema'] == 'experimental-codebase-ir-metadata-manifest@2','actual metadata schema')
    require(mapped(manifest['output']) == directory,'exact retained native shard output')
    same(manifest['limits'],NATIVE_METADATA_LIMITS,'unchanged complete original native metadata limits')
    require(manifest['source_snapshot_sha256'] == digest(manifest['source_snapshot']), 'metadata source snapshot identity')
    false(manifest['qualification'],('production_activated','admitted','proof_authority'))
    full_rows, bounded = {}, {}
    for family, descriptor in manifest['families'].items():
        require(re.fullmatch('[a-z][a-z0-9_]{0,39}',family) is not None,'safe complete family')
        raw = pin(descriptor['export'],directory)
        values = [parse(line) for line in raw.splitlines()]
        require(raw == b''.join(wire(value)+b'\n' for value in values)
                and type(descriptor['count']) is int and len(values) == descriptor['count']
                and descriptor['digest'] == digest(values), 'complete canonical family export')
        for ordinal,value in enumerate(values):
            require(set(value) == {'schema','family','row_id','payload_sha256','source_snapshot_sha256','row_ordinal','payload'}
                    and value['schema'] == 'experimental-codebase-ir-metadata-row@1' and value['family'] == family
                    and type(value['row_ordinal']) is int and value['row_ordinal'] == ordinal
                    and value['payload_sha256'] == digest(value['payload'])
                    and value['source_snapshot_sha256'] == manifest['source_snapshot_sha256'], 'exact full metadata row')
            identity = {key:value[key] for key in ('schema','family','payload_sha256','source_snapshot_sha256')}
            require(value['row_id'] == digest(identity),'metadata row identity')
            key = family,value['row_id']
            require(key not in full_rows,'unique metadata row')
            full_rows[key] = value
        bounded[family] = [value['payload'] for value in values]
    require(manifest['row_count'] == len(full_rows) and manifest['row_root_sha256'] == digest({
        family:[value for (name,_),value in full_rows.items() if name == family] for family in bounded}),
        'complete metadata count and row root')
    with database(directory/'metadata.duckdb') as connection:
        tables = connection.execute('SELECT schema_name,table_name,sql FROM duckdb_tables() WHERE NOT internal ORDER BY schema_name,table_name').fetchall()
        views = connection.execute('SELECT schema_name,view_name,sql FROM duckdb_views() WHERE NOT internal ORDER BY schema_name,view_name').fetchall()
        require(manifest['catalog_schema_sha256'] == digest({'tables':[list(row) for row in tables],
                'views':[list(row) for row in views]}),'actual read-only native catalog schema digest')
        observed = connection.execute('SELECT family,row_id,payload_sha256,source_snapshot_sha256,row_ordinal,payload_json FROM metadata_rows ORDER BY family,row_ordinal').fetchall()
        require(len(observed) == len(full_rows),'complete read-only SQL row population')
        for family,key,ph,sh,ordinal,payload in observed:
            value = full_rows[family,key]
            same([ph,sh,ordinal,payload],[value['payload_sha256'],value['source_snapshot_sha256'],value['row_ordinal'],wire(value['payload']).decode()],
                 'complete SQL/export bytes correspondence')
        same(connection.execute('SELECT singleton,manifest_json FROM metadata_identity').fetchall(),
             [(1,wire(manifest).decode())], 'SQL exact manifest')
    lake = defaultdict(list)
    with database(directory/'lake/history.ducklake',parquet=True) as connection:
        snapshots = [row[0] for row in connection.execute('SELECT snapshot_id FROM ducklake_snapshot ORDER BY snapshot_id').fetchall()]
        require(snapshots == list(range(len(manifest['batches'])+2)),'complete isolated DuckLake snapshot chain')
        require(connection.execute('SELECT count(*) FROM ducklake_delete_file').fetchone()[0] == 0,'append-only retained history')
        tables = {key:(name,path) for key,name,path in connection.execute('SELECT table_id,table_name,path FROM ducklake_table').fetchall()}
        file_count = 0
        for table,path,relative,count,size,end in connection.execute('SELECT table_id,path,path_is_relative,record_count,file_size_bytes,end_snapshot FROM ducklake_data_file ORDER BY data_file_id').fetchall():
            require(relative is True and end is None,'retained current relative Parquet file')
            name,table_path = tables[table]
            file = inside(directory/'lake/data/main'/table_path/path,directory/'lake/data/main')
            before = read(file)
            require(len(before) == size,'actual Parquet byte size')
            cursor = connection.execute('SELECT * FROM read_parquet(?)',[str(file)])
            names = [column[0] for column in cursor.description]
            values = [dict(zip(names,row)) for row in cursor.fetchall()]
            require(len(values) == count and read(file) == before,'read-only core Parquet complete population and fence')
            lake[name].extend(values)
            file_count += 1
    same({name:len(values) for name,values in lake.items()}, {'identity':1,'sources':1,'commits':len(manifest['batches']),
            'events':manifest['lake_packet_count']},'complete native lake table populations')
    batches = {}
    for descriptor in manifest['batches']:
        raw = pin(descriptor,directory)
        batch = parse(raw)
        require(wire(batch) == raw and batch['batch_id'] not in batches,'canonical distinct native batch')
        batches[batch['batch_id']] = batch
        false(descriptor['receipt'],('production_activated','admitted'))
    for row in lake['commits']:
        same(parse(row['batch_json']),batches[row['batch_id']],'exact Parquet/batch commit')
    packet_rows, event_ids = {}, set()
    for row in lake['events']:
        event = parse(row['event_json'])
        packet = event['payload']
        require(packet['schema'] == 'experimental-codebase-ir-metadata-row-packet@1'
                and packet['packet_id'] == digest({k:v for k,v in packet.items() if k != 'packet_id'})
                and row['event_id'] == event['event_id'] == packet['packet_id']
                and event['kind'] == 'codebase_ir_'+packet['family'],'complete native event/packet identities')
        raw = packet['payload_json'].encode()
        values = parse(raw)
        require(raw == wire(values) and packet['payload_sha256'] == 'sha256:'+sha(raw)
                and len(raw) == packet['payload_bytes'] and len(values) == packet['row_count']
                and packet['ordered_row_ids'] == [value['row_id'] for value in values], 'complete canonical packet bytes')
        require(event['event_id'] not in event_ids,'unique lake event')
        event_ids.add(event['event_id'])
        for value in values:
            key = value['family'],value['row_id']
            require(key not in packet_rows,'unique row packet membership')
            same(value,full_rows[key],'complete packet/export/SQL row correspondence')
            packet_rows[key] = value
    require(set(packet_rows) == set(full_rows),'all complete native rows recovered from Parquet')
    require({event['event_id'] for batch in batches.values() for event in batch['events']} == event_ids
            and manifest['lake_snapshot_digest'] == digest([item['receipt'] for item in manifest['batches']]),
            'all actual batch events and snapshot receipts')
    return bounded,manifest,{'rows':len(full_rows),'families':len(bounded),
                          'packets':len(event_ids),'parquet_files':file_count,'snapshots':len(snapshots)}


def native_readback_report(manifest,raw):
    return {'schema':'experimental-codebase-ir-metadata-readback@2','output':manifest['output'],
        'manifest_sha256':'sha256:'+sha(raw),'source_snapshot_sha256':manifest['source_snapshot_sha256'],
        'row_count':manifest['row_count'],'row_root_sha256':manifest['row_root_sha256'],
        'family_counts':{name:value['count'] for name,value in manifest['families'].items()},
        'family_digests':{name:value['digest'] for name,value in manifest['families'].items()},
        'family_views':{name:value['view'] for name,value in manifest['families'].items()},
        'exports':{name:value['export'] for name,value in manifest['families'].items()},
        'native_runtime':manifest['native_runtime'],'limits':manifest['limits'],
        'lake_layout':manifest['lake_layout'],'lake_packet_count':manifest['lake_packet_count'],
        'lake_snapshot_ids':[value['receipt']['snapshot_id'] for value in manifest['batches']],
        'lake_snapshot_digest':manifest['lake_snapshot_digest'],
        'lake_receipts':[value['receipt'] for value in manifest['batches']],
        'qualification':{'experimental':True,'production_activated':False,'admitted':False,'proof_authority':False}}


def metadata_audit(helper):
    """Independently join bounded native shards to one complete producer file."""
    directory = ADVISORY/'metadata'
    manifest_raw = read(directory/'manifest.json',16*1024**2)
    manifest = parse(manifest_raw)
    require(set(manifest) == {'schema','output','source_snapshot','source_snapshot_sha256','source_snapshot_artifact',
        'input_families','limits','native_limits','chunk_policy','packaged_sha256','producer_sha256',
        'row_count','payload_bytes','family_counts','family_digests','row_root_sha256','qualification','parts','exports'}
        and wire(manifest) == manifest_raw, 'closed canonical global shard manifest')
    require(manifest['schema'] == 'experimental-codebase-ir-sharded-metadata-manifest@1',
            'explicit additive sharded profile required')
    require(mapped(manifest['output']) == directory,'exact retained global metadata assembly output')
    same(manifest['limits'],SHARDED_METADATA_LIMITS,'explicit additive global/part bounds')
    same(manifest['native_limits'],NATIVE_METADATA_LIMITS,'all original native metadata bounds remain unchanged')
    same(manifest['chunk_policy'],{'schema':'terminal-codebase-metadata-chunk@1','bytes':49152,'family':'metadata_chunks',
                                  'producer_artifact_bytes':32000000},
         'unchanged complete producer chunk policy')
    same(manifest['qualification'],{'experimental':True,'production_activated':False,'admitted':False,
        'proof_authority':False,'execution_authority':False,'completion_authority':False,
        'whole_population_native_transaction':False}, 'closed assembly-only authority scope')
    source_raw = pin(manifest['source_snapshot_artifact'],directory)
    require(source_raw == wire(manifest['source_snapshot']) and len(source_raw) <= 1048576,
            'physical complete global source snapshot under original cap')
    require(manifest['source_snapshot_sha256'] == digest(manifest['source_snapshot']),
            'global exact source snapshot identity')
    input_families = manifest['input_families']
    require(type(input_families) is list and input_families == sorted(set(input_families))
            and all(re.fullmatch('[a-z][a-z0-9_]{0,39}',name) is not None for name in input_families),
            'closed sorted original input family population')
    require({'ast','kg','vectors','contracts'} <= set(input_families) and len(input_families) <= 32,
            'joined fixture retains all native default producer families')
    require(all(set(manifest[field]) == set(input_families) for field in ('family_counts','family_digests','exports')),
            'exact global family map populations')
    assembled,bounded,global_rows = {name:[] for name in input_families},{},{}
    counters = {name:0 for name in input_families}
    part_summaries = []
    require(type(manifest['parts']) is list and 1 <= len(manifest['parts']) <= 16,
            'bounded complete shard population')
    require(sorted(path.name for path in (directory/'parts').iterdir()) ==
            [format(ordinal,'06d') for ordinal in range(len(manifest['parts']))],
            'no omitted or extraneous shard namespace')
    for ordinal,part in enumerate(manifest['parts']):
        require(type(part['ordinal']) is int and part['ordinal'] == ordinal
                and part['relative_path'] == 'parts/'+format(ordinal,'06d'),
                'contiguous canonical native shard identities')
        part_directory = inside(directory/part['relative_path'],directory/'parts')
        require(set(part) == {'ordinal','relative_path','row_count','payload_bytes','global_family_ranges',
                             'source_snapshot_sha256','native_manifest_sha256','native_manifest','native_report'}
                and part['native_manifest']['relative_path'] == part['relative_path']+'/manifest.json',
                'closed exact native part manifest descriptor')
        raw = pin(part['native_manifest'],directory)
        require(sha(raw) == part['native_manifest_sha256'].removeprefix('sha256:'),
                'whole actual native shard manifest digest')
        values,native,summary = native_metadata_audit(part_directory)
        same(part['native_report'],native_readback_report(native,raw),
             'complete per-shard native readback report reconstructed independently')
        require(set(values) == set(input_families), 'complete shard family population')
        require(native['row_count'] == part['row_count']
                and native['source_snapshot_sha256'] == part['source_snapshot_sha256'],
                'actual shard native population and source identity')
        source = native['source_snapshot']
        require(source['schema'] == 'experimental-codebase-ir-sharded-part-source@1'
                and type(source['part_ordinal']) is int and source['part_ordinal'] == ordinal
                and source['global_source_snapshot_sha256'] == manifest['source_snapshot_sha256']
                and source['global_packaged_sha256'] == manifest['packaged_sha256']
                and source['global_producer_sha256'] == manifest['producer_sha256'],
                'exact shard/global identity binding')
        same(source['original_source_snapshot'],manifest['source_snapshot'],
             'complete original global source snapshot retained in each shard')
        same(source['global_family_ranges'],part['global_family_ranges'],
             'native source binds exact global ranges')
        require(set(part['global_family_ranges']) == set(input_families),
                'ranges bind every original family including empty slices')
        payload_bytes = 0
        for family in input_families:
            interval = part['global_family_ranges'][family]
            require(set(interval) == {'start','stop','count'}
                    and all(type(interval[key]) is int for key in interval)
                    and interval['start'] == counters[family]
                    and interval['stop'] >= interval['start']
                    and interval['count'] == interval['stop']-interval['start'] == len(values[family]),
                    'exact complete contiguous global family slice')
            counters[family] = interval['stop']
            assembled[family].extend(values[family])
            payload_bytes += sum(len(wire(value)) for value in values[family])
        require(type(part['payload_bytes']) is int and payload_bytes == part['payload_bytes'] and payload_bytes <= 32*1024**2
                and type(part['row_count']) is int and part['row_count'] <= 32768,
                'each actual native shard obeys additive tighter part bounds')
        same(native['limits'],manifest['native_limits'],'all original native limits retained unchanged')
        summary.update({'ordinal':ordinal,'payload_bytes':payload_bytes,
                        'native_manifest_sha256':sha(raw)})
        part_summaries.append(summary)
    require(set(manifest['exports']) == set(input_families),'one complete global export per original family')
    for family in input_families:
        export = manifest['exports'][family]
        require(export['relative_path'] == 'exports/'+family+'.jsonl' and type(export['bytes']) is int
                and 0 <= export['bytes'] <= 256*1024**2,'closed bounded global export path')
        raw = pin(export,directory)
        values = [parse(line) for line in raw.splitlines()]
        require(raw == b''.join(wire(value)+b'\n' for value in values),
                'canonical complete global assembly export')
        bounded[family] = []
        for ordinal,value in enumerate(values):
            require(set(value) == {'schema','family','row_id','payload_sha256','source_snapshot_sha256','row_ordinal','payload'}
                    and value['schema'] == 'experimental-codebase-ir-metadata-row@1'
                    and value['family'] == family and type(value['row_ordinal']) is int and value['row_ordinal'] == ordinal
                    and value['payload_sha256'] == digest(value['payload'])
                    and value['source_snapshot_sha256'] == manifest['source_snapshot_sha256'],
                    'exact global assembly row with global ordinal/source')
            require(value['row_id'] == digest({key:value[key] for key in ('schema','family','payload_sha256','source_snapshot_sha256')}),
                    'complete global assembly row identity')
            bounded[family].append(value['payload'])
        require(len(values) == counters[family] == manifest['family_counts'][family],
                'global exports equal complete native shard range populations')
        require(type(manifest['family_counts'][family]) is int
                and manifest['family_digests'][family] == digest(bounded[family]),
                'complete original payload-array family digest')
        same(bounded[family],assembled[family],'whole global export/native shard payload correspondence')
        global_rows[family] = values
    require(manifest['packaged_sha256'] == digest(bounded),'entire assembled packaged payload digest')
    require(type(manifest['row_count']) is int and 0 <= manifest['row_count'] <= 524288
            and manifest['row_count'] == sum(map(len,bounded.values()))
            and type(manifest['payload_bytes']) is int and 0 <= manifest['payload_bytes'] <= 512*1024**2
            and manifest['payload_bytes'] == sum(len(wire(value)) for values in bounded.values() for value in values)
            and manifest['row_root_sha256'] == digest(global_rows),
            'complete global row/byte accounting')
    positions,starts,expected_parts = {name:0 for name in input_families},{name:0 for name in input_families},[]
    pending_rows = pending_bytes = 0
    def flush():
        nonlocal starts,pending_rows,pending_bytes
        expected_parts.append({'row_count':pending_rows,'payload_bytes':pending_bytes,
            'global_family_ranges':{name:{'start':starts[name],'stop':positions[name],
                'count':positions[name]-starts[name]} for name in input_families}})
        starts,pending_rows,pending_bytes = dict(positions),0,0
    for family in input_families:
        for value in bounded[family]:
            size = len(wire(value))
            if pending_rows and (pending_rows == 32768 or pending_bytes+size > 32*1024**2):
                flush()
            positions[family] += 1
            pending_rows += 1
            pending_bytes += size
    if pending_rows or not expected_parts:
        flush()
    same(expected_parts,[{key:part[key] for key in ('row_count','payload_bytes','global_family_ranges')}
                         for part in manifest['parts']], 'independent exact deterministic native partition boundaries')
    complete = reconstruct(bounded)
    require(manifest['producer_sha256'] == digest(complete)
            == manifest['source_snapshot']['producer_sha256'], 'whole reconstructed original producer identity')
    producer_raw = read(ADVISORY/'complete-producer-records.json',256*1024**2)
    producer = parse(producer_raw)
    require(producer_raw == wire(producer)+b'\n', 'canonical independently retained complete producer payload')
    same(producer,complete,'entire independently retained original producer equals all reconstructed native shards')
    same({name:len(values) for name,values in complete.items()},helper['complete_family_counts'],
         'all complete original producer family populations')
    same({name:len(values) for name,values in bounded.items()},helper['packaged_family_counts'],
         'all complete packaged family populations')
    report_keys = ('output','source_snapshot_sha256','input_families','packaged_sha256','producer_sha256',
                  'row_count','payload_bytes','row_root_sha256','family_counts','family_digests',
                  'exports','parts','limits','native_limits','chunk_policy','qualification')
    readback = {'schema':'experimental-codebase-ir-sharded-metadata-readback@1',
                'manifest_sha256':'sha256:'+sha(manifest_raw),**{key:manifest[key] for key in report_keys}}
    for label in ('metadata','metadata_replay'):
        same({key:value for key,value in helper[label].items() if key != 'fresh_process_readback'},
             readback,'complete retained global native readback result '+label)
    fresh = helper['metadata_replay']['fresh_process_readback']
    require(fresh['verified'] is True and fresh['global_new_process_verified'] is True
            and fresh['method'] == 'new_python_process_native_part_validation_and_global_reassembly'
            and fresh['manifest_sha256'] == readback['manifest_sha256']
            and fresh['packaged_sha256'] == manifest['packaged_sha256']
            and fresh['producer_sha256'] == manifest['producer_sha256']
            and fresh['row_count'] == manifest['row_count'], 'historical actual global new-process readback receipt')
    plain = {}
    for family,values in complete.items():
        require(all(set(value) == {'schema','occurrence','record'}
                    and value['schema'] == 'finite-advisory-worker-metadata-occurrence@1'
                    and type(value['occurrence']) is int and value['occurrence'] == ordinal
                    for ordinal,value in enumerate(values)), 'all complete ordered producer occurrence wrappers')
        plain[family] = [value['record'] for value in values]
    return plain,manifest,{'schema':manifest['schema'],'parts':part_summaries,
        'rows':manifest['row_count'],'payload_bytes':manifest['payload_bytes'],
        'families':len(input_families),'complete_family_counts':helper['complete_family_counts'],
        'complete_producer_bytes':len(producer_raw),'complete_producer_sha256':sha(producer_raw),
        'packaged_sha256':manifest['packaged_sha256'],'producer_sha256':manifest['producer_sha256'],
        'global_sql_store_claimed':False}


def source_audit(result,metadata,plain):
    cas = {}
    for file in sorted((NATIVE/'cas').rglob('*')):
        if file.is_file():
            raw = read(file)
            name = file.name
            require((raw_cid(raw) if name.startswith('bafk') else cid(parse(raw))) == name,'actual physical content store identity')
            require(name not in cas or read(cas[name]) == raw,'consistent duplicate CAS bytes')
            cas[name] = file
    heads = [result['original_head'],result['successor_head']]
    require(all(type(head['generation']) is int for head in heads)
            and [head['generation'] for head in heads] == [1,2]
            and heads[0]['repository_id'] == heads[1]['repository_id']
            and heads[0]['snapshot_cid'] != heads[1]['snapshot_cid'],'actual native successor generation')
    same(metadata['source_snapshot']['heads'],heads,'metadata complete original/successor head binding')
    source_maps,commits = [],[]
    for head in heads:
        manifest = load(cas[head['manifest_cid']])
        require(cid(manifest) == head['manifest_cid'] and manifest['ast_revision_id'] == head['ast_revision_id']
                and manifest['snapshot']['snapshot_cid'] == head['snapshot_cid'],'native full head manifest')
        entries = {entry['path']:entry for entry in manifest['snapshot']['entries']}
        require(set(entries) == PATHS and len(manifest['units']) == 10,'complete ten source/AST capture')
        for entry in entries.values():
            raw = read(cas[entry['source_cid']])
            require(len(raw) == entry['size_bytes'] and raw_cid(raw) == entry['source_cid'],'captured exact source CAS')
        for unit in manifest['units']:
            require(unit['parse_status'] == 'ok' and cid(load(cas[unit['ast_cid']])) == unit['ast_cid'],'complete captured AST CAS')
        require(manifest['semantic_state']['symbols'] and manifest['semantic_state']['edges'],'nonempty captured source graph')
        source_maps.append(entries)
        commits.append(manifest['snapshot']['git_commit'])
    require(commits == [result['original_commit'],result['published_commit']],'capture follows actual publication Git heads')
    require(sorted(path for path in PATHS if source_maps[0][path]['source_cid'] != source_maps[1][path]['source_cid']) == ['calc.py']
            and read(cas[source_maps[0]['calc.py']['source_cid']]) == BEFORE
            and read(cas[source_maps[1]['calc.py']['source_cid']]) == AFTER,'only reviewed exact increment source changes')
    for path,entry in source_maps[1].items():
        require(read(NATIVE/'repository'/path) == read(cas[entry['source_cid']]),'complete published repository/CAS source')
    require({file.relative_to(NATIVE/'repository').as_posix() for file in (NATIVE/'repository').rglob('*')
             if file.is_file() and not file.is_relative_to(NATIVE/'repository/.git')} == PATHS,'complete closed authored source worktree')
    publications = load(NATIVE/'native-source-publications.json')
    for name,head in zip(('initial','successor'),heads):
        require(cid(publications[name]) == head['receipt_cid'],'raw native publication receipt identity')
        same({key:publications[name][key] for key in ('repository_id','generation','manifest_cid','snapshot_cid','ast_revision_id')},
             {key:head[key] for key in ('repository_id','generation','manifest_cid','snapshot_cid','ast_revision_id')},'complete publication/head binding')
    require(publications['initial']['previous_head'] is None,'initial publication ancestry')
    same(publications['successor']['previous_head'],heads[0],'native expected-head successor CAS')
    with database(NATIVE/'codebase.duckdb') as connection:
        expected = tuple(heads[1][key] for key in ('repository_id','generation','manifest_cid','snapshot_cid','ast_revision_id','receipt_cid'))
        require(connection.execute('SELECT repository_id,generation,manifest_cid,snapshot_cid,ast_revision_id,receipt_cid FROM codebase_control.heads').fetchall() == [expected], 'actual durable native current head')
        actual = connection.execute('SELECT revision_id,path,source_cid FROM source_files').fetchall()
        expected = {(head['ast_revision_id'],path):entry['source_cid'] for head,entries in zip(heads,source_maps) for path,entry in entries.items()}
        require(len(actual) == 20 and {(revision,path):source for revision,path,source in actual} == expected,'both complete durable source revisions')
        for source,ast_cid,payload in connection.execute('SELECT source_cid,ast_cid,payload_json FROM ast_blobs').fetchall():
            same(parse(payload),load(cas[ast_cid]),'actual AST SQL/CAS full payload')
        invalidations = rows(connection,'invalidations')
        same(ordered(invalidations),ordered(publications['invalidations']),'complete durable invalidation records')
        require(any(row['reason'] == 'revision_superseded' for row in invalidations),'actual native successor invalidation')
    require(len(plain['sources']) >= 40 and len(plain['ast']) >= 40 and
            {'imports','calls'} <= {row['relation'] for row in plain['kg']}, 'all complete captures and import/call graph metadata')
    dispositions = [row for row in plain['compiled_logic'] if row.get('schema') == 'finite-reviewed-candidate-native-unit-disposition@1']
    require(len(dispositions) >= 40,'complete frontend dispositions across all captures')
    for row in dispositions:
        require(row['disposition'] == 'retained_unproved_native_frontend'
                and row['solver_executed'] is False and row['source_semantics_verified'] is False
                and row['proof_authority'] is False,'unproved frontend disposition scope')
    return heads,cas,source_maps


def training_audit(helper,heads):
    contexts = {prefix+'-'+mode:load(ADVISORY/(prefix+'-'+mode+'-context')/'context.json')
                for prefix in ('root','child') for mode in ('off','training','frozen')}
    root_cp = load(ADVISORY/'root-training-context/checkpoint.json')
    child_cp = load(ADVISORY/'child-training-context/checkpoint.json')
    same(root_cp['contract'],child_cp['contract'],'exact frozen modality contract')
    same(root_cp['feature_space'],child_cp['feature_space'],'exact frozen source feature basis')
    same(root_cp['state']['optimizer_config'],child_cp['state']['optimizer_config'],'exact resumed Adam configuration')
    modality_identity = {'canonicalization':'ir-canonical-json-v1','collection_semantics':{},
        'domain':'autoencoder.modality','identity_profile':'ir-canonical-identity-v1',
        'payload':root_cp['contract'],'schema_version':'autoencoder-modality-contract/v1'}
    strict_plain(modality_identity)
    contract_sha256 = sha(wire(modality_identity))
    require({item['native_profile_id'] for item in root_cp['contract']['projections']} == {'program_ir','dynamic_hoare'}
            and all(item['family_id'] == 'program' for item in root_cp['contract']['projections']),
            'actual trained source-bound native structural projection families')
    require(child_cp['report']['base_state_sha256'] == sha(wire(root_cp['state'],ascii=True)), 'exact retained base Adam state digest')
    role_policy = {'calc.py':'train','known_variant.py':'train','tune.py':'tune','canary.py':'canary'}
    for label,checkpoint,head in (('root',root_cp,heads[0]),('child',child_cp,heads[1])):
        report = checkpoint['report']
        provenance = report['codebase_provenance']
        require(type(report['attempted_epochs']) is int and report['attempted_epochs'] == 16 and len(report['epochs']) == 16,'actual retained sixteen attempted epochs')
        same({key:report['configuration'][key] for key in ('epochs','learning_rate','seed','latent_width')},
             {'epochs':16,'learning_rate':.01,'seed':1729,'latent_width':8},'predeclared trainer settings')
        require({item['path']:item['role'] for item in provenance['selections']} == role_policy
                and all(item['contracts'] == [] for item in provenance['selections']),'exact four selection roles without behavioral teachers')
        same(provenance['head'],head,'actual source generation bound to checkpoint')
        require(checkpoint['state']['latent_width'] == 8 and len(checkpoint['state']['adam']) == 4,'actual bounded model width and Adam tensors')
        require(all(type(item['step']) is int and item['step'] == checkpoint['state']['completed_epochs'] for item in checkpoint['state']['adam']),'selected optimizer-step lineage')
        best_loss,best_metrics = report['before']['objective'],report['before']['projections']
        for epoch in report['epochs']:
            require(all(type(epoch[key]) in (int,float) and math.isfinite(epoch[key]) for key in ('train_objective','tuning_objective','gradient_norm')),'finite actual epoch telemetry')
            metrics = epoch['projection_metrics']
            selected = (not epoch['deadline_exceeded'] and epoch['tuning_objective'] < best_loss and
                        all(metrics[name][key] <= best_metrics[name][key]+1e-9 for name in metrics for key in ('reconstruction','cosine')))
            require(epoch['selected'] is selected,'independent retained projection-wise checkpoint selection')
            if selected:
                best_loss,best_metrics = epoch['tuning_objective'],metrics
        same(report['after'],{'objective':best_loss,'projections':best_metrics},'complete selected held-tuning state')
        require(report['after']['objective'] <= report['before']['objective'] and provenance['canary_monitoring']['used_for_selection'] is False,'finite tuning nonincrease and diagnostic canary exclusion')
        false(provenance,('source_runtime_semantics_verified','behavioral_satisfaction','proof_authority','completion_authority','admission_authority','promotion_performed'))
    for field in ('tuning_targets','canary_targets','replay_targets'):
        same(root_cp['report']['codebase_provenance'][field],child_cp['report']['codebase_provenance'][field],'fixed cohort '+field)
    require(child_cp['report']['codebase_provenance']['continuation'] == 'exact_frozen_basis_adam_resume'
            and child_cp['report']['before']['objective'] == root_cp['report']['after']['objective'],'actual resumed held-tuning baseline')
    for name,context in contexts.items():
        directory = ADVISORY/(name+'-context')
        self_cid(context,'context_cid')
        expected_head = heads[1 if name.startswith('child-') else 0]
        same(context['head'],expected_head,'exact retained source-bound context')
        expected_mode = 'model_off' if name.endswith('-off') else 'train' if name.endswith('-training') else 'frozen'
        require(context['mode'] == expected_mode and type(context['actual_training_delta']) is int
                and context['actual_training_delta'] == (16 if expected_mode == 'train' else 0),'training versus exact no-fit context')
        require(all(value is False for value in context['authority'].values()),'learned context authority remains false')
        for descriptor in context['artifacts'].values():
            pin(descriptor,directory)
            body = {key:value for key,value in descriptor.items() if key != 'blob_cid'}
            require(descriptor['blob_cid'] == cid(body),'full context blob descriptor identity')
        if expected_mode == 'model_off':
            continue
        record = load(directory/'record.json')
        checkpoint = load(directory/'checkpoint.json')
        inference = load(directory/'inference.json')
        require(record['state_sha256'] == sha(wire(checkpoint['state'],ascii=True))
                and record['feature_space_sha256'] == sha(wire(checkpoint['feature_space'],ascii=True))
                and record['contract_sha256'] == context['contract_sha256'] == checkpoint['state']['contract_sha256'] == contract_sha256
                and context['model_record_cid'] == cid(record), 'complete model/contract/numerical/feature identities')
        require(inference['training_executed'] is False and inference['inference']['decoded_formulas_generated'] is False
                and all(len(row['latent']) == 8 for row in inference['inference']['rows']),'full vectors and no formal decoder claim')
        if name.startswith('child-'):
            lineage = load(directory/'lineage.json')
            require(len(lineage) == 2,'complete exact root/child ancestry')
            same(lineage[1]['checkpoint'],root_cp,'immutable exact retained root checkpoint ancestor')
    inventory = {}
    with database(ADVISORY/'train.duckdb') as connection:
        names = [row[0] for row in connection.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='autoencoder_control' ORDER BY table_name").fetchall()]
        for name in names:
            require(re.fullmatch('[a-z_]+',name) is not None,'native registry table name')
            values = sorted([list(row) for row in connection.execute('SELECT * FROM autoencoder_control.'+name).fetchall()],key=repr)
            inventory[name] = {'row_count':len(values),'rows_sha256':digest(values)}
        require(connection.execute('SELECT owner_generation FROM autoencoder_control.meta').fetchone()[0] == 2
                and inventory['versions']['row_count'] == 2 and inventory['heads']['row_count'] == 0,'two actual versions/owner generations without promotion')
        versions = {version:{'version_id':version,'variant_id':variant,'parent_version_id':parent,'artifact':parse(artifact),'metadata':parse(metadata)}
                    for version,variant,parent,artifact,metadata in connection.execute('SELECT version_id,variant_id,parent_version_id,artifact,metadata FROM autoencoder_control.versions').fetchall()}
        require(set(versions) == {contexts['root-training']['version_id'],contexts['child-training']['version_id']},'exact two retained version IDs')
        for name in ('root-training','child-training'):
            for item in load(ADVISORY/(name+'-context')/'lineage.json'):
                same(item['version'],versions[item['version']['version_id']],'direct SQL complete immutable model version')
                descriptor = item['version']['artifact']
                file = ADVISORY/'model-artifacts'/descriptor['sha256'][:2]/descriptor['sha256']
                raw = read(file)
                require(len(raw) == descriptor['bytes'] and sha(raw) == descriptor['sha256'],'physical registry immutable checkpoint artifact')
                same(parse(raw),item['checkpoint'],'whole lineage checkpoint physical bytes')
    same(helper['cold_verification']['registry_after']['tables'],inventory,'actual cold retained no-fit registry inventory')
    costs = load(ADVISORY/'phase-costs.json')['training_cost']
    require(costs['known_completed_context_epochs'] == costs['checkpoint_observed_epochs'] == costs['observed_epoch_lower_bound'] == 32
            and len(costs['checkpoints']) == 2,'exact completed root+child attempted-training cost')
    for checkpoint in costs['checkpoints']:
        require(checkpoint['attempted_epochs'] == 16 and checkpoint['worker_receipt']['returncode'] == 0,'actual retained trainer child receipts')
        raw = pin(checkpoint['artifact'])
        report = parse(raw)['report']
        require(checkpoint['request_sha256'] == report['codebase_request_sha256'] and report['attempted_epochs'] == 16,'unique actual checkpoint request digest')
    return contexts,{'versions':2,'contexts':6,'attempted_epochs':32,
                    'root_before':root_cp['report']['before']['objective'],'root_after':root_cp['report']['after']['objective'],
                    'child_before':child_cp['report']['before']['objective'],'child_after':child_cp['report']['after']['objective'],
                    'native_projection_profiles':['dynamic_hoare','program_ir'],
                    'convergence_proved':False,'generalization_verified':False}


def lowering_audit(heads,cas,source_maps):
    summary = []
    for ordinal,name in enumerate(('before-lowering.json','successor-lowering.json')):
        value = load(ADVISORY/name)
        require(value['profile'] == 'guarded-python-integer-offset-ast-lowering@2'
                and value['scope'] == 'formal_ast_lowering_correctness','explicit defined-grammar lowering profile')
        false(value,LOWERING_FALSE)
        require(all(value[field] is True for field in ('source_ast_semantics_defined','source_ast_lowering_proved','native_target_correspondence_proved','kernel_checked_model')),'historical exact grammar/kernel-check scope')
        require(value['requested_model_theorem_proved'] is (ordinal == 1)
                and value['status'] == ('model_proved' if ordinal else 'model_refuted'),'requested offset proof/refutation partition')
        same(value['head'],heads[ordinal],'proof exact native generation')
        raw = owned_artifacts(value)
        source = BEFORE if ordinal == 0 else AFTER
        require(raw['source'] == source and value['source_cid'] == raw_cid(source)
                and value['source_sha256'] == sha(source),'proof binds exact captured Python source')
        contract = value['contract']
        require(contract['path'] == 'calc.py' and contract['function_name'] == 'increment' and contract['parameter'] == 'n'
                and type(contract['offset']) is int and contract['offset'] == 2 and cid(contract) == value['contract_cid'],'exact original desired contract')
        translation = value['translation']
        same(parse(raw['translation']),translation,'full retained translation bytes')
        require(cid(translation) == value['translation_cid'],'translation full identity')
        module = ast.parse(source.decode('ascii'),type_comments=True)
        require(len(module.body) == 1 and type(module.body[0]) is ast.FunctionDef
                and module.body[0].name == 'increment' and len(module.body[0].body) == 1
                and type(module.body[0].body[0]) is ast.Return,'closed exact independent Python AST')
        expression = module.body[0].body[0].value
        require(type(expression) is ast.BinOp and type(expression.op) is ast.Add
                and type(expression.left) is ast.Name and expression.left.id == 'n'
                and type(expression.right) is ast.Constant and type(expression.right.value) is int
                and expression.right.value == ordinal+1,'independent exact integer-offset AST')
        body = {'kind':'add','literal':{'kind':'literal','value':ordinal+1}}
        target = {'kind':'add','left':{'kind':'parameter'},'right':{'kind':'literal','value':ordinal+1}}
        same(translation['source_ast_syntax'],body,'exact source grammar translation')
        same(translation['lowered_source_syntax'],target,'independent mathematical lowering')
        same(translation['native_target_syntax'],target,'native target syntax binding')
        program = translation['native_program']
        require(translation['native_program_cid'] == cid(program),'complete actual native ProgramIR root')
        expressions = {row['expression_id']:row for row in program['expressions']}
        parameter = program['functions'][0]['parameter_symbol_ids'][0]
        def target_of(key):
            row = expressions[key]
            if row['kind'] == 'symbol':
                require(row['symbol_ids'] == [parameter],'exact native parameter symbol')
                return {'kind':'parameter'}
            if row['kind'] == 'literal':
                require(type(row['attributes']['value']) is int,'native integer literal')
                return {'kind':'literal','value':row['attributes']['value']}
            require(row['kind'] == 'binary' and row['operator'] == 'add' and len(row['operand_ids']) == 2,'actual native addition expression')
            return {'kind':'add','left':target_of(row['operand_ids'][0]),'right':target_of(row['operand_ids'][1])}
        same(target_of(program['commands'][0]['expression_ids'][0]),target,'independent executable native target correspondence')
        same(parse(raw['manifest']),load(cas[heads[ordinal]['manifest_cid']]),'full owned lowering/native manifest')
        entry = source_maps[ordinal]['calc.py']
        manifest = parse(raw['manifest'])
        ast_cid = next(unit['ast_cid'] for unit in manifest['units'] if unit['entry_cid'] == entry['entry_cid'])
        same(parse(raw['source_ast']),load(cas[ast_cid]),'exact captured AST full physical CAS')
        certificate = value['lean_certificate']
        same(parse(raw['lean_certificate']),certificate,'whole retained historical Lean certificate')
        require(certificate['translation_cid'] == value['translation_cid']
                and certificate['source_cid'] == raw_cid(raw['lean_source'])
                and certificate['olean_cid'] == raw_cid(raw['lean_olean'])
                and certificate['process_policy_cid'] == value['process_policy_cid'],'all Lean certificate artifact roots')
        for process in (certificate['version_process'],certificate['process']):
            successful_process(process)
            require(process['command'][0] == value['tool_policy']['lean']['path']
                    and process['limits']['max_output_bytes'] == 1024*1024
                    and 1 <= process['limits']['timeout_ms'] <= 90000
                    and process['limits']['resident_memory_bytes'] == 512*1024**2,'actual per-child Lean bounded profile')
        require(certificate['version_process']['command'][1:] == ['--version']
                and certificate['process']['command'][1:] == ['-j','1','-o','IntegerLowering.olean','IntegerLowering.lean'],'actual retained Lean version/compile argv')
        lean = raw['lean_source'].decode()
        require(not re.search(r'\b(sorry|admit|axiom)\b',lean),'no generated Lean theorem placeholder')
        match = re.search(r'def translationEvidence : String := ("(?:[^"\\]|\\.)*")',lean)
        require(match is not None,'Lean source embeds exact translation evidence')
        same(parse(parse(match.group(1))),translation,'Lean exact embedded mathematical translation')
        theorems = ['signed_literal_lowering_correct','lowering_correct','native_target_correspondence',
                    'captured_ast_target_equivalence','source_offset_identity'] + (['requested_offset_identity'] if ordinal else ['requested_offset_counterexample','requested_goal_refuted'])
        require(certificate['theorems'] == theorems and all('theorem '+name+' ' in lean for name in theorems)
                and '∀ body : SourceBody, ∀ input : Int' in lean,'actual generic grammar and concrete target theorem sources')
        summary.append({'generation':ordinal+1,'status':value['status'],'source_offset':ordinal+1,
                        'requested_offset':2,'olean_sha256':sha(raw['lean_olean']),'theorems':theorems,
                        'kernel_reexecution_performed':False})
    return summary


def admission_audit(heads):
    admissions = [load(NATIVE/name) for name in ('before-admission.json','successor-admission.json','cold-admission.json')]
    declarations,receipts = [],[]
    for ordinal,admission in enumerate(admissions):
        require(set(admission) == {'declaration','graph','evidence','local_admission','receipt'},'closed complete finite admission')
        binding = {key:admission['declaration']['binding'][key] for key in ('identity','profile_id')}
        declaration = public_signature(admission['declaration'],binding)
        manifest = public_signature(declaration['manifest'],binding)
        receipt = public_signature(admission['receipt'],binding)
        declarations.append(declaration)
        receipts.append(receipt)
        require({task['task_key'] for task in manifest['tasks']} == {'FINITE-TYPE','FINITE-OFFSET'}
                and len(manifest['tasks']) == len(admission['graph']['tasks']) == 2,'all signed administrator tasks retained')
        require(receipt['declaration_cid'] == cid(admission['declaration'])
                and receipt['graph_cid'] == cid(admission['graph'])
                and receipt['evidence_cid'] == cid(admission['evidence']),'signed complete admission roots')
        false(receipt,FINITE_FALSE)
        semantic = receipt['semantic_context']
        false(semantic,FINITE_FALSE)
        require(receipt['semantic_context_cid'] == cid(semantic) and semantic['task_population_preserved'] is True
                and semantic['finite_facts_are_context_only'] is True and len(semantic['administrator_task_cids']) == 2
                and set(semantic['native_task_bindings']) == {'finite-integer-type-goal','finite-integer-offset-goal'},'closed two-clause context without task omission')
        evidence = admission['evidence']
        self_cid(evidence,'result_cid')
        match = evidence['match']
        query = match['query']
        self_cid(query,'query_cid')
        same(query,semantic['query'],'entire signed native IntentIR finite query')
        require(query['requirement_ids'] == ['finite-integer-offset-goal','finite-integer-type-goal']
                and {row['statement_id'] for row in query['statements']} == set(query['requirement_ids'])
                and query['contract']['offset'] == 2 and type(query['contract']['offset']) is int,'exact original offset/type statements and contract')
        expected = 1 if ordinal == 0 else 2
        observation = match['observation']
        require(evidence['current_facts_count'] == len(match['current_facts']) == expected
                and match['domain_inputs'] == observation['domain_inputs'] == [-2,-1,0,1,2]
                and observation['observations'] == [{'input':n,'input_type':'int','output':n+expected,'output_type':'int'} for n in [-2,-1,0,1,2]],'all five actual finite observations and facts')
        require(observation['type_clause_satisfied'] is True and observation['offset_clause_satisfied'] is (ordinal > 0)
                and observation['runtime_observation_coverage_complete'] is True,'exact finite clause outcomes')
        require(evidence['selected_task_ids'] == (['task:finite:offset'] if ordinal == 0 else [])
                and semantic['residual_requirement_ids'] == (['finite-integer-offset-goal'] if ordinal == 0 else [])
                and receipt['planning_permitted'] is (ordinal == 0)
                and receipt['no_work_review_only'] is (ordinal > 0),'exact residual versus no-work admission partition')
        for producer in (observation,evidence['operational_model']):
            artifacts = owned_artifacts(producer)
            require(artifacts['source'] == (BEFORE if ordinal == 0 else AFTER),'finite producers exact source bytes')
            successful_process(producer['lean_certificate']['process'])
            require(producer['lean_certificate']['olean_cid'] == raw_cid(artifacts['lean_olean'])
                    and not re.search(r'\b(sorry|admit|axiom)\b',artifacts['lean_source'].decode()),'finite retained Lean source/olean correspondence')
        self_cid(evidence['source_custody'],'custody_cid')
        require(set(evidence['source_custody']['admitted_source_paths']) == PATHS,'complete ten-source finite custody')
        if ordinal < 2:
            same(declaration['head'],heads[ordinal],'exact signed original/successor head')
        if ordinal == 0:
            require(admission['local_admission'] is not None,'initial full guarded local admission')
            local = admission['local_admission']
            same(local['manifest'],declaration['manifest'],'exact original local manifest')
            public_signature(local['manifest'],binding)
            local_receipt = public_signature(local['receipt'],binding)
            require(local_receipt['manifest_cid'] == cid(local['manifest']),'signed local manifest identity')
        else:
            require(admission['local_admission'] is None,'no current execution grant from no-work facts')
    for declaration in declarations[1:]:
        same(declaration['manifest']['payload']['tasks'],declarations[0]['manifest']['payload']['tasks'],'unchanged exact original administrator task specifications')
        same(declaration['task_bindings'],declarations[0]['task_bindings'],'unchanged two original native task bindings')
    same(admissions[0]['graph']['tasks'],admissions[1]['graph']['tasks'],'complete same original graph task objects after publication')
    for prefix,expected in (('root',admissions[0]['evidence']),('child',admissions[1]['evidence'])):
        baseline = None
        for mode in ('off','train','frozen'):
            preview = load(ADVISORY/(prefix+'-'+mode+'-preview.json'))
            false(preview,('source_semantics_verified','runtime_behavior_verified','proof_authority','execution_authority','completion_authority','mutation_authority','omission_authority','worker_launched','convergence_proved','production_admitted'))
            require(preview['training_steps_during_preview'] == 0 and preview['reservation_released_on_return'] is True,'no fitting or retained lease in preview')
            match = preview['match']
            projection = {key:match[key] for key in ('source_cid','query','domain_inputs','clause_results','eligible_clause_ids','residual_clause_ids')}
            projection['observations'] = match['observation']['observations']
            projection['selected_task_ids'] = preview['selected_task_ids']
            projection['current_facts_count'] = preview['current_facts_count']
            if baseline is None:
                baseline = projection
            same(projection,baseline,'entire two-clause semantic projection across off/train/frozen')
            same(match['query'],expected['match']['query'],'complete preview query equals original signed intent')
            same(match['clause_results'],expected['match']['clause_results'],'complete preview outcomes equal signed current clauses')
    return admissions,declarations,receipts


def candidate_audit(admission,contexts,heads):
    reviewed = load(ADVISORY/'reviewed-candidate.json')
    payload = public_signature(reviewed,{key:admission['declaration']['binding'][key] for key in ('identity','profile_id')})
    generated = load(ADVISORY/'generated-candidate.json')
    bridge = load(ADVISORY/'worker-bridge.json')
    descriptor = load(NATIVE/'candidate-descriptor.json')
    worker_raw = pin({'path':descriptor['artifact'],'bytes':mapped(descriptor['artifact']).stat().st_size,'sha256':descriptor['sha256']})
    worker = parse(worker_raw)
    require(worker_raw == wire(worker),'canonical full public worker handoff')
    self_cid(worker,'candidate_cid')
    self_cid(bridge,'bridge_cid')
    false(payload,CANDIDATE_FALSE)
    false(generated,CANDIDATE_FALSE)
    false(bridge,CLAIMS)
    false(worker,('source_semantics_verified','proof_authority','execution_authority','publication_authority','completion_authority','task_omission_authority'))
    require(payload['parent_admission_cid'] == generated['parent_admission_cid'] == bridge['admission_cid']
            == worker['finite_admission_cid'] == descriptor['finite_admission_cid'] == cid(admission),'same whole original signed parent admission')
    same(generated['parent_admission'],admission,'whole signed parent retained by generator')
    same(worker['finite_admission'],admission,'whole signed parent retained by actual worker handoff')
    same(generated['reviewed_candidate'],reviewed,'whole reviewed signed candidate retained by generator')
    require(generated['reviewed_candidate_cid'] == bridge['reviewed_candidate_cid'] == cid(reviewed)
            and bridge['generated_result_cid'] == generated['result_cid']
            and bridge['worker_candidate_cid'] == worker['candidate_cid'] == descriptor['candidate_cid'],'complete exact review/generation/worker identities')
    for value in (payload,generated,bridge):
        same(value['head'],heads[0],'proposal original native generation')
    require(payload['task_cid'] == generated['task_cid'] == bridge['task_cid'] == worker['task_cid'] == descriptor['task_cid']
            and worker['task_id'] == 'FINITE-OFFSET' and payload['task_key'] == 'FINITE-OFFSET'
            and type(worker['task_revision']) is int and worker['task_revision'] == bridge['task_revision'] == descriptor['task_revision'],'same exact residual task and revision')
    semantic = admission['receipt']['payload']['semantic_context']
    same(bridge['administrator_task_cids'],semantic['administrator_task_cids'],'both original task CIDs retained in bridge')
    require(worker['semantic_context_cid'] == admission['receipt']['payload']['semantic_context_cid']
            and worker['original_prompt'] == admission['declaration']['payload']['source_text']
            and worker['original_clause_ids'] == semantic['query']['requirement_ids']
            and worker['operation_catalog_cid'] == semantic['operation_catalog_cid'],'exact prompt, IntentIR clauses and operation catalog')
    require(payload['path'] == 'calc.py' and payload['function_name'] == 'increment' and payload['parameter'] == 'n'
            and type(payload['desired_offset']) is int and payload['desired_offset'] == 2
            and base64.b64decode(payload['before_base64'],validate=True) == BEFORE
            and base64.b64decode(payload['after_base64'],validate=True) == AFTER,'exact separately reviewed single offset change')
    same(worker['edit'],{'path':'calc.py','effect':'modify','before_bytes_base64':base64.b64encode(BEFORE).decode(),
        'after_bytes_base64':base64.b64encode(AFTER).decode(),'before_sha256':sha(BEFORE),'after_sha256':sha(AFTER)},'exact actual worker edit with no extra output/change')
    raw = owned_artifacts(generated)
    require(raw['source'] == BEFORE and raw['replacement'] == AFTER and parse(raw['candidate']) == reviewed,'exact trusted proposal bytes')
    require(payload['implementation']['driver_sha256'] == sha(raw['driver']) and payload['implementation']['driver_bytes'] == len(raw['driver']),'signed trusted driver bytes')
    for source in payload['implementation']['source_pins'].values():
        observed = pin(source)
        require(raw_cid(observed) == source['cid'],'selected source producer raw bytes')
    process = generated['child_process']
    same(parse(raw['process']),process,'whole actual trusted subprocess receipt')
    successful_process(process)
    require(type(process['pid']) is int and process['pid'] > 0
            and process['command'] == [payload['python']['path'],'-I','-S','driver.py',sha(raw['candidate'])]
            and process['stdout'] == '' and process['limits']['max_output_bytes'] == 65536
            and type(process['effective_timeout_ms']) is int and 1 <= process['effective_timeout_ms'] <= 5000
            and process['elapsed_ms'] <= 5000,'bounded exact trusted proposal process')
    child = parse(raw['child_receipt'])
    require(child['pid'] == process['pid'] and type(child['uid']) is int and child['candidate_sha256'] == sha(raw['candidate'])
            and child['before_sha256'] == sha(BEFORE) and child['after_sha256'] == sha(AFTER)
            and child['task_cid'] == payload['task_cid'],'trusted child result joins exact process/source/task')
    same(process['limits'],payload['limits'],'signed exact child resource profile')
    require(len(generated['native_capacity']['observations']) == 4,'four actual held native resource observations')
    reservations = [item['reservation'] for item in generated['native_capacity']['observations']]
    require([item['memory_mb'] for item in reservations] == [1024,512,512,1024]
            and reservations[1]['parent_lease_id'] == reservations[0]['lease_id']
            and reservations[0]['lease_id'] == reservations[3]['lease_id']
            and reservations[1]['lease_id'] == reservations[2]['lease_id'],'actual root/child lease lineage')
    require(generated['canonical_source_unchanged'] is generated['reservation_released_on_return'] is True
            and generated['proposal_generated'] is True,'generator observation preserves canonical source and releases reservation')
    same(bridge['model_binding'],contexts['root-training'],'complete retained separately advisory root model binding')
    require(bridge['training_steps'] == bridge['provider_calls'] == 0 and worker['training_steps'] == worker['provider_calls'] == 0,'no fitting or provider calls in candidate/native handoff')
    return bridge,worker,generated,{'trusted_proposal_pid':process['pid'],'trusted_proposal_uid':child['uid'],
                                   'worker_candidate_cid':worker['candidate_cid'],'proposal_result_cid':generated['result_cid'],
                                   'python_sha256':payload['python']['sha256'],
                                   'same_uid_proposal_isolation_claimed':False,'process_origin_attested':False}


def git_read(repository,*arguments):
    env = {'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','GIT_CONFIG_NOSYSTEM':'1',
           'GIT_CONFIG_GLOBAL':'/dev/null','GIT_OPTIONAL_LOCKS':'0'}
    process = subprocess.run(['/usr/bin/git','-c','safe.directory='+str(repository),'-C',str(repository),*arguments],
                             capture_output=True,env=env,timeout=30)
    require(process.returncode == 0 and len(process.stdout)+len(process.stderr) <= 2*1024**2,'bounded read-only Git object inspection')
    return process.stdout


def native_worker_audit(result,admission,worker,source_maps,cas):
    export = load(NATIVE/'native-task-evidence.json')
    require(export['schema'] == 'finite-advisory-worker-native-sql-evidence@1','complete native SQL export schema')
    require(set(export['tables']) == {'objectives','goals','plans','tasks','task_dependencies','task_outputs',
        'task_validations','task_acceptance','task_attempts','task_claims','leases','fencing_epochs','validation_runs',
        'validation_results','completion_receipts','merge_attempts'},'all sixteen selected native SQL projections')
    with database(NATIVE/'private/intent.duckdb') as connection:
        actual = {table:rows(connection,table) for table in export['tables']}
        for table,expected in export['tables'].items():
            same(ordered(actual[table]),ordered(expected),'complete direct SQL/export correspondence: '+table)
        events = rows(connection,'domain_events')
    materialized = load(NATIVE/'materialized.json')
    task_cids = set(materialized['task_cids'])
    tasks = {row['task_cid']:row for row in actual['tasks']}
    require(len(tasks) == 2 and set(tasks) == task_cids
            and {row['task_alias']:row['status'] for row in tasks.values()} == {'FINITE-TYPE':'completed','FINITE-OFFSET':'completed'}
            and all(type(row['revision']) is int and row['revision'] == 4 for row in tasks.values()),'exact complete original tasks completed at revision four')
    require(len(actual['plans']) == len(actual['goals']) == len(actual['objectives']) == 1,'one original full-population objective/goal/plan')
    same(load(NATIVE/'completed-task-rows.json'),result['completed_task_rows'],'whole retained completed task projection')
    require(set(result['task_rows_before_candidate']) == set(result['completed_task_rows']) == task_cids,'both original tasks retained through proposal/publication')
    plan = parse(actual['plans'][0]['body_json'])
    ref = plan['finite_repository_admission_ref']
    retained = pin(ref)
    require(cid(parse(retained)) == ref['admission_cid'] == cid(admission),'complete historical admission retained in actual native plan')
    same(parse(retained),admission,'original native plan whole signed parent')
    expected_receipts = {}
    local_validations = []
    portal_rows = []
    signed_publication_native_bodies = []
    for row in actual['completion_receipts']:
        require(row['task_cid'] in tasks and row['task_cid'] not in expected_receipts,'one current-revision completion per original task')
        body = parse(row['body_json'])
        require(body['schema'] == 'ipfs_accelerate_py/agent-supervisor/intent-completion-evidence@1'
                and type(body['revision']) is int and body['revision'] == 4,'current complete evidence schema')
        receipt = body['receipt']
        same(receipt,parse(tasks[row['task_cid']]['body_json'])['completion_receipt'],'actual task/completion receipt complete body equality')
        evidence = cid({'task_cid':row['task_cid'],'revision':4,'receipt':receipt,'evidence_digests':body['evidence_digests']})
        require(evidence == row['evidence_digest'] and row['receipt_cid'] == cid({'namespace':'completion-receipt',
                'task_cid':row['task_cid'],'revision':4,'evidence_digest':evidence}),'current complete native evidence/receipt roots')
        require(receipt['operation'] == 'database_complete' and type(receipt['fencing_token']) is int
                and receipt['fencing_token'] == receipt['fence_epoch'] == 1
                and all(type(receipt[key]) is str and receipt[key] for key in ('attempt_id','claim_id','lease_id','owner_session_id')),'actual native claim/lease/fence receipt lineage')
        expected_receipts[row['task_cid']] = receipt
    require(set(expected_receipts) == task_cids,'both current original completion receipts')
    for row in actual['validation_results']:
        require(row['task_cid'] in tasks and row['outcome'] == 'passed','actual passed public native validation row')
        body = parse(row['body_json'])
        run = next(run for run in actual['validation_runs'] if run['run_id'] == row['run_id'])
        require(run['status'] == 'passed' and run['task_cid'] == row['task_cid']
                and run['attempt_id'] == expected_receipts[row['task_cid']]['attempt_id'],'public native validation current attempt join')
        if 'local_observed_validation' in body:
            # The original local_completion_missing gate requires this event
            # join for signed public checks. Native Portal acceptance uses its
            # executor validation/preparation/receipt route checked below.
            require(any(event['event_type'] == 'intent.validation_recorded' and event['task_cid'] == row['task_cid']
                        and parse(event['body_json'])['subject_id'] == row['result_id'] for event in events),
                    'actual domain-event/signed-public-check join')
            signed = body['local_observed_validation']
            signer = {key:admission['declaration']['binding'][key] for key in ('identity','profile_id')}
            checked = public_signature(signed,signer)
            task = tasks[row['task_cid']]
            contract = parse(task['body_json'])['local_planning_contract']
            public_signature(contract,signer)
            manifest = public_signature(contract['payload']['manifest'],signer)
            require(contract['payload']['manifest_cid'] == cid(contract['payload']['manifest'])
                    and contract['payload']['pending_cid'] == cid(contract['payload']['pending_requirements'])
                    and contract['payload']['task_cid'] == row['task_cid']
                    and contract['payload']['task_key'] == task['task_alias']
                    and contract['payload']['task_spec'] in manifest['tasks']
                    and all(item['phase'] == 'post_execution' and item['required'] is True
                            and item['kind'] in ('test','review') and item['minimum_code_assurance'] == 'candidate'
                            for item in contract['payload']['pending_requirements']),
                    'complete signed original post-execution public guard contract')
            require(contract['payload']['task_spec']['validations'] == [checked['validation']],
                    'exact complete original public validation specification')
            require(checked['task_cid'] == row['task_cid'] and type(checked['task_revision']) is int and checked['task_revision'] == 3
                    and checked['attempt_id'] == run['attempt_id'] and checked['contract_cid'] == cid(contract)
                    and checked['manifest_cid'] == materialized['manifest_cid'] and checked['pending_cid'] == materialized['pending_cid']
                    and checked['intent_owner_id'] == contract['payload']['intent_owner_id']
                    and checked['outcome'] == 'passed' and type(checked['exit_code']) is int and checked['exit_code'] == 0
                    and row['evidence_digest'] == cid(signed),'owner-signed exact current task/contract/public validation')
            if task['task_alias'] == 'FINITE-OFFSET':
                transition = public_signature(checked['source_transition'],signer)
                require(transition['task_cid'] == row['task_cid'] and transition['task_revision'] == 3
                        and transition['attempt_id'] == run['attempt_id']
                        and transition['published_commit'] == result['published_commit']
                        and transition['baseline_commit'] == result['original_commit']
                        and transition['contract_cid'] == cid(contract)
                        and transition['manifest_cid'] == materialized['manifest_cid']
                        and transition['completion_authority'] is False
                        and transition['changed_paths'] == ['calc.py'], 'owner-signed exact accepted source transition')
                same(transition['task_spec'],contract['payload']['task_spec'], 'published transition retains exact original public task specification')
                signed_publication_native_bodies.append(transition['native_transition'])
                sources = transition['sources']
                expected_sources = source_maps[1]
            else:
                require('source_transition' not in checked, 'prerequisite validation retains its original baseline')
                sources = manifest['sources']
                expected_sources = source_maps[0]
            require(set(sources) == PATHS and checked['source_tree_id'] == cid({
                    'schema':'supervisor-local-source-tree@1','sources':sources}),
                    'public check binds the exact complete signed source tree')
            for path,entry in expected_sources.items():
                same(sources[path],{'executable':False,'sha256':sha(read(cas[entry['source_cid']]))},
                     'public check source tree matches complete captured original/published bytes')
            local_validations.append(checked)
        else:
            require(body['validator'] == 'DatabasePortalExecutionBridge@1' and body['outcome'] == 'passed'
                    and body['task_cid'] == row['task_cid'] and body['attempt_id'] == run['attempt_id']
                    and body['evidence_digest'] == row['evidence_digest'],'actual Portal validation evidence join')
            portal_rows.append(body)
    require(len(local_validations) == 2 and len(portal_rows) == 1 and len(actual['validation_runs']) == 3,'two original public checks plus one actual Portal acceptance')
    offset = next(task for task in tasks.values() if task['task_alias'] == 'FINITE-OFFSET')
    receipt = expected_receipts[offset['task_cid']]
    preparation = receipt['coordination_preparation']
    require(preparation['control_expected_revision'] == 3 and preparation['control_expected_status'] == 'in_progress'
            and preparation['status'] == 'prepared' and preparation['replayed'] is False,'actual native completion preparation gate')
    for key in ('attempt_id','claim_id','lease_id','owner_session_id','fencing_token','fence_epoch','evidence_digest'):
        same(preparation[key],receipt[key],'same actual prepared completion lineage '+key)
    validation = preparation['body']['validation']
    same(validation,receipt['validation'],'exact prepared/committed Portal validation')
    same(validation,portal_rows[0],'exact native persisted Portal acceptance')
    transition = validation['accepted_source_transition']
    require(len(signed_publication_native_bodies) == 1,'one signed public publication guard')
    same(signed_publication_native_bodies[0],transition,
         'owner-signed public source guard and native Portal acceptance bind the same whole transition')
    require(transition['schema'] == 'ipfs_accelerate_py/agent-supervisor/accepted-source-transition@1'
            and transition['authority'] == 'database_completion_cas_after_portal_and_git_verification'
            and transition['task_completion_authority'] is transition['worker_self_approval'] is False
            and transition['database_task_cid'] == offset['task_cid'] and transition['task_alias'] == 'FINITE-OFFSET'
            and transition['baseline_ref'] == result['original_commit']
            and transition['implementation_commit'] == result['published_commit_parents'][1]
            and transition['merge_commit'] == result['published_commit'],'native Portal accepted exact baseline→implementation→merge transition')
    binding = transition['database_attempt_binding']
    require(binding['schema'] == 'ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@2'
            and binding['interface'] == 'DatabasePortalExecutionBridge@1' and binding['projection_authority'] is False
            and binding['task_cid'] == offset['task_cid'] and binding['goal_cid'] == offset['goal_cid']
            and binding['plan_cid'] == offset['plan_cid'] and binding['task_alias'] == 'FINITE-OFFSET'
            and type(binding['task_revision']) is int and binding['task_revision'] == binding['control_expected_revision'] == 3,'exact original database task/Portal projection binding')
    for key in ('attempt_id','claim_id','lease_id','fencing_token','fence_epoch'):
        same(binding[key],receipt[key],'actual Portal/native claim binding '+key)
    portal = load(NATIVE/'native-portal-binding.json')
    same(parse(pin(portal['artifact'])),portal['record'],'raw actual retained Portal binding')
    same(portal['record'],binding,'actual launch Portal binding matches completion body')
    proof = transition['integration_commit_proof']
    require(proof['passed'] is True and proof['reasons'] == []
            and proof['implementation_commit'] == result['published_commit_parents'][1]
            and proof['integration_commit'] == proof['integration_ref'] == result['published_commit'],'actual accepted integration commit proof')
    repository = NATIVE/'repository'
    commit = result['published_commit']
    parents = git_read(repository,'rev-list','--parents','-n','1',commit).decode().split()
    require(parents == [commit,*result['published_commit_parents']] and len(parents) == 3
            and parents[1] == result['original_commit'],'independent actual Git merge ancestry')
    require(git_read(repository,'diff','--name-only',result['original_commit'],commit).decode().splitlines() == ['calc.py']
            and git_read(repository,'show',result['original_commit']+':calc.py') == BEFORE
            and git_read(repository,'show',commit+':calc.py') == AFTER
            and git_read(repository,'status','--porcelain') == b'','actual exact published one-output merge and clean source')
    scope = public_signature(load(NATIVE/'execution-scope.json'),{key:admission['declaration']['binding'][key] for key in ('identity','profile_id')})
    require(scope['finite_admission_cid'] == cid(admission),'signed native scope exact original parent')
    same(scope['candidate']['descriptor'],load(NATIVE/'candidate-descriptor.json'),'signed actual launch descriptor')
    require(scope['candidate']['descriptor']['candidate_cid'] == worker['candidate_cid'],'actual launch consumes retained verified proposal')
    stop = result['stop']
    cleanup = stop['data']['isolated_worker_cleanup']
    require(stop['status'] == 'succeeded' and stop['data']['old_tree_fenced'] is True
            and cleanup['worker_uid'] == 1001 and type(cleanup['returncode']) is int and cleanup['returncode'] == 0
            and cleanup['single_worker'] is True and cleanup['completion_authority'] is False
            and type(result['remaining_processes']) is int and result['remaining_processes'] == 0
            and result['bootstrap_errors'] == [],'actual STOP/fenced isolated worker and zero observed processes')
    owner_cleanup = load(NATIVE/'owner-fixture-worktree-cleanup.json')
    require(type(owner_cleanup) is list and owner_cleanup and all(
            value['scope'] == 'explicit owner fixture cleanup after native STOP; not native completion recovery'
            and value['decision']['allowed'] is True and value['allocation']['task_id'] == 'FINITE-OFFSET'
            for value in owner_cleanup),'separate explicit owner worktree cleanup evidence')
    observed_allocations = list(result['observed_worker_allocations'].values()) if type(result['observed_worker_allocations']) is dict else result['observed_worker_allocations']
    for value in owner_cleanup:
        require(any(wire(value['allocation']) == wire(allocation) for allocation in observed_allocations),'owner cleanup refers to actual observed native allocation')
    require(not (repository/'.git/worktrees').exists(),'all linked fixture worktrees removed after STOP')
    with database(NATIVE/'private/no-work.duckdb') as connection:
        require(all(connection.execute('SELECT count(*) FROM '+table).fetchone()[0] == 0 for table in ('tasks','plans','goals','objectives')),'no-work materialization creates no task graph or omission grant')
    return {'task_count':2,'completed_revisions':[4,4],'public_checks':2,'portal_acceptances':1,
            'signed_public_checks_with_required_intent_events':2,
            'portal_acceptance_bound_to_prepared_and_committed_receipt':True,
            'typed_receipt_columns_not_used_as_claim_authority':True,'worker_uid':1001,'published_commit':commit,
            'published_commit_parents':parents[1:],'changed_paths':['calc.py'],'stop_returncode':0,
            'owner_fixture_cleanup_is_crash_recovery':False}


def advisory_original_artifact_audit(value, archive, contexts, lowering, generated, descriptor, cas):
    """Join original pins to independently audited producer bytes, not just seals.

    Copied host inodes are never evidence about original container identities.
    The live registry is intentionally mutable; only disposable paired copies of
    its sealed DB/WAL bytes are opened, with a read-only DuckDB connection.
    """
    archived = {row['original']['path']:row for row in archive['files']}
    pins = {row['path']:row for row in value['files']}
    require(set(archived) == set(pins), 'complete original artifact archive population')
    expected = {}
    def add(path, raw=None, *, size=None, checksum=None):
        require(type(path) is str and path in pins, 'independent mandatory artifact missing: '+str(path))
        if raw is not None:
            size,checksum = len(raw),sha(raw)
        require(type(size) is int and 0 <= size <= 64*1024**2
                and type(checksum) is str and re.fullmatch('[0-9a-f]{64}',checksum),
                'independent finite mandatory expected identity')
        wanted = (size,checksum)
        require(path not in expected or expected[path] == wanted, 'conflicting independent mandatory identity: '+path)
        expected[path] = wanted
        original = pins[path]
        require((original['size_bytes'],original['sha256']) == wanted,
                'signed mandatory artifact differs from its actual producer: '+path)
        raw_archive = pin(archived[path]['snapshot'])
        require((len(raw_archive),sha(raw_archive)) == wanted
                and (raw is None or raw_archive == raw),
                'archived mandatory bytes differ from actual immutable producer: '+path)
    def container_path(path):
        path = inside(path)
        return '/results/'+str(path.relative_to(ROOT))
    def add_cas(identity, raw):
        require(identity in cas, 'mandatory immutable CAS mirror missing: '+identity)
        path = cas[identity]
        require(read(path) == raw, 'mandatory CAS mirror exact original producer bytes')
        add(container_path(path),raw)
    for name in ('root-training','root-frozen'):
        context = contexts[name]
        base = '/results/native/private/advisory/'+name+'-context'
        add(base+'/context.json',wire(context))
        require(set(context['artifacts']) == {'checkpoint','record','lineage','metrics','inference','invocation'},
                'complete native selected model artifact roles')
        for artifact in context['artifacts'].values():
            raw = pin(artifact, ADVISORY/(name+'-context'))
            add(base+'/'+artifact['relative_path'],raw)
        record = load(ADVISORY/(name+'-context')/'record.json')
        require(cid(record) == context['model_record_cid'], 'root actual model record CAS identity')
        add_cas(context['model_record_cid'],wire(record))
        lineage = load(ADVISORY/(name+'-context')/'lineage.json')
        require(len(lineage) == 1 and lineage[0]['version']['version_id'] == context['version_id'],
                'exact root-only immutable checkpoint ancestry at launch')
        for ancestor in lineage:
            artifact = ancestor['version']['artifact']
            path = '/results/native/private/advisory/model-artifacts/'+artifact['sha256'][:2]+'/'+artifact['sha256']
            raw = read(mapped(path))
            require(len(raw) == artifact['bytes'] and sha(raw) == artifact['sha256'],
                    'actual selected registry immutable checkpoint bytes')
            same(parse(raw),ancestor['checkpoint'],'complete selected immutable checkpoint ancestry bytes')
            add(path,raw)
    lower_raw = wire(lowering)
    add(lowering['output']+'/result.json',lower_raw)
    add_cas(cid(lowering),lower_raw)
    add_cas(raw_cid(lower_raw),lower_raw)
    for artifact in lowering['artifacts'].values():
        raw = pin(artifact)
        require(raw_cid(raw) == artifact['cid'], 'lowering artifact exact raw identity')
        add(artifact['path'],raw)
        add_cas(artifact['cid'],raw)
    for name in ('python','lean'):
        seal = lowering['tool_policy'][name]
        add(seal['path'],size=seal['size_bytes'],checksum=seal['sha256'])
    add(generated['output']+'/result.json',wire(generated))
    for artifact in generated['artifacts'].values():
        raw = pin(artifact)
        require(raw_cid(raw) == artifact['cid'], 'generated candidate exact artifact identity')
        add(artifact['path'],raw)
    handoff = read(mapped(descriptor['artifact']))
    require(sha(handoff) == descriptor['sha256'], 'independent full public worker handoff bytes')
    add(descriptor['artifact'],handoff)
    manifest = load(cas[value['head']['manifest_cid']])
    add_cas(value['head']['manifest_cid'],wire(manifest))
    for entry in manifest['snapshot']['entries']:
        original = read(cas[entry['source_cid']])
        require(len(original) == entry['size_bytes'] and raw_cid(original) == entry['source_cid'],
                'original source CAS length and content identity')
        # calc.py now contains the published successor. Its original working
        # bytes are joined to original Git/CAS, never to its successor contents.
        add('/results/native/repository/'+entry['path'],original)
        add_cas(entry['source_cid'],original)
    for unit in manifest['units']:
        if unit['ast_cid'] is not None:
            raw = read(cas[unit['ast_cid']])
            require(cid(parse(raw)) == unit['ast_cid'], 'exact original AST CAS root')
            add_cas(unit['ast_cid'],raw)
    root_checkpoint = load(ADVISORY/'root-training-context/checkpoint.json')
    implementations = dict(root_checkpoint['report']['codebase_provenance']['implementation']['files'])
    implementations.update(lowering['translation']['frontend']['source_sha256'])
    core = (
        'ipfs_accelerate_py.agent_supervisor.runtime.finite_advisory_artifact_closure',
        'ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_admission',
        'ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate',
        'ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate_runner',
        'ipfs_accelerate_py.agent_supervisor.planning.codebase_feature_context',
        'ipfs_accelerate_py.agent_supervisor.planning.finite_integer_source_custody',
        'ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission',
        'ipfs_datasets_py.logic.software_contracts.codebase_integer_lowering_lean',
        'ipfs_datasets_py.logic.software_contracts.codebase_source_training')
    for name in core:
        prefix = '/opt/ipfs-supervisor/source/' if name.startswith('ipfs_accelerate_py.') else '/opt/ipfs-supervisor/datasets/'
        path = prefix+name.replace('.','/')+'.py'
        actual = sha(read(mapped(path)))
        require(name not in implementations or implementations[name] == actual,
                'actual selected native producer agrees with immutable producer provenance')
        implementations[name] = actual
    same(value['implementation'],implementations, 'exact independently selected producer module population')
    for name,checksum in implementations.items():
        prefix = '/opt/ipfs-supervisor/source/' if name.startswith('ipfs_accelerate_py.') else '/opt/ipfs-supervisor/datasets/'
        path = prefix+name.replace('.','/')+'.py'
        raw = read(mapped(path))
        require(sha(raw) == checksum, 'actual retained producer source bytes match selected provenance')
        add(path,raw)
    registry_path = '/results/native/private/advisory/train.duckdb'
    wal_path = registry_path+'.wal'
    registry = archived[registry_path]
    registry_raw = pin(registry['snapshot'])
    wal_raw = pin(archived[wal_path]['snapshot']) if wal_path in archived else None
    root_observation = load(ADVISORY/'root-frozen-preview-artifacts-registry-inventory.json')
    require(root_observation['unchanged'] is True and root_observation['fitting_performed'] is False
            and root_observation['promotion_performed'] is False, 'actual prelaunch no-fit registry observation')
    same(root_observation['before'],root_observation['after'], 'root prelaunch registry no-fit complete population')
    inventory = {}
    # The original source namespace remains read only. A read-only connection
    # applies an exactly paired archived WAL to disposable copies in memory;
    # no owner, native extension, fit, proof, promotion, or SQL mutation runs.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='finite-advisory-registry-audit-') as temporary:
        replay = Path(temporary)/'train.duckdb'
        replay.write_bytes(registry_raw)
        if wal_raw is not None:
            Path(str(replay)+'.wal').write_bytes(wal_raw)
        DATABASES.append({'scope':'disposable original launch DB/WAL paired replay',
                          'source':registry['snapshot']['path'], 'wal_present':wal_raw is not None,
                          'read_only':True,'external_access':False})
        with duckdb.connect(str(replay),read_only=True,config={
                'threads':'1','enable_external_access':'false','autoload_known_extensions':'false',
                'autoinstall_known_extensions':'false'}) as connection:
            names = [row[0] for row in connection.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='autoencoder_control' ORDER BY table_name").fetchall()]
            require(set(names) == {'events','heads','meta','operations','outbox','runs','variants','versions'},
                    'complete original launch root registry table population')
            for name in names:
                require(re.fullmatch('[a-z_]+',name) is not None, 'closed root registry table name')
                values = sorted([list(row) for row in connection.execute('SELECT * FROM autoencoder_control.'+name).fetchall()],key=repr)
                inventory[name] = {'row_count':len(values),'rows_sha256':digest(values)}
            same(inventory,root_observation['before']['tables'], 'archived DB/WAL exact original prelaunch root registry inventory')
            require(connection.execute('SELECT owner_generation FROM autoencoder_control.meta').fetchone()[0]
                    == root_observation['before']['owner_generation'] == 1
                    and inventory['versions']['row_count'] == inventory['variants']['row_count'] == 1
                    and inventory['heads']['row_count'] == inventory['runs']['row_count'] == 0,
                    'original root model only, no child generation, run or promotion at launch')
            versions = [{'version_id':version,'variant_id':variant,'parent_version_id':parent,
                        'artifact':parse(artifact),'metadata':parse(metadata)}
                        for version,variant,parent,artifact,metadata in connection.execute(
                            'SELECT version_id,variant_id,parent_version_id,artifact,metadata FROM autoencoder_control.versions').fetchall()]
            lineage = load(ADVISORY/'root-training-context/lineage.json')
            same(versions,[lineage[0]['version']], 'archive SQL exact selected immutable root version')
        require(replay.read_bytes() == registry_raw and
                (wal_raw is None or Path(str(replay)+'.wal').read_bytes() == wal_raw),
                'disposable read-only DB/WAL replay preserves exact copied input bytes')
    require(pin(registry['snapshot']) == registry_raw
            and (wal_raw is None or pin(archived[wal_path]['snapshot']) == wal_raw),
            'original archive DB/WAL unchanged by independent read-only replay')
    same(root_observation['before']['artifacts'],[
        {'bytes':item['version']['artifact']['bytes'],
         'sha256':item['version']['artifact']['sha256'],
         'path':'/results/native/private/advisory/model-artifacts/'+item['version']['artifact']['sha256'][:2]+'/'+item['version']['artifact']['sha256']}
        for item in load(ADVISORY/'root-training-context/lineage.json')],
        'prelaunch root artifact inventory joins actual immutable selected ancestry')
    # Extra preparation artifacts are independently joined to their retained
    # producer files too; DB/WAL are the only intentionally mutable exceptions.
    for path in pins:
        if path not in expected and path not in {registry_path,wal_path}:
            add(path,read(mapped(path)))
    return {'independent_original_artifact_count':len(expected),
            'mutable_registry_files':1+int(wal_raw is not None),
            'registry_db_wal_paired_read_only_replay':True,
            'registry_wal_present':wal_raw is not None,
            'registry_query_scope':'read-only original DB plus exact paired WAL when present, on disposable copies',
            'original_registry_owner_generation':1,'original_registry_version_count':1,
            'copied_original_os_witnesses_used_as_live_authority':False}


def advisory_historical_runtime_grant_audit(path, envelope, admission, signer, result):
    """Verify the actual retained runtime grant, strictly as historical bytes."""
    grant = load(path)
    require(type(grant) is dict and set(grant) == {'manifest','signature'}, 'closed actual runtime launch grant')
    manifest = public_signature({'payload':grant['manifest'],'binding':grant['signature']},signer)
    require(set(manifest) == {'schema','repository_root','state_root','repository_id','run_id',
        'baseline_commit','baseline_tree_id','local_admission','owner_identity','execution_route_policy_id',
        'argv','argv_digest','environment','bootstrap_listener_inode','bootstrap_client_id',
        'provider_dispatch_allowed','coordination_attempt_safety_cap','lifetime_seconds','worker_worktree_root',
        'candidate_runner','task_context_bundle','context_refresh_policy','published_retrieval_policy',
        'production_activation','completion_authority','finite_execution_scope','finite_worker_launcher'},
        'closed actual native runtime launch manifest')
    require(manifest['schema'] == 'isolated-admitted-supervisor-launch@1'
            and manifest['production_activation'] is manifest['completion_authority'] is False,
            'historical native launch grant profile and authority')
    same(manifest['finite_execution_scope'],envelope,'actual runtime grant signs the entire exact execution scope')
    require(manifest['provider_dispatch_allowed'] is True
            and type(manifest['coordination_attempt_safety_cap']) is int
            and manifest['coordination_attempt_safety_cap'] == 1
            and type(manifest['lifetime_seconds']) is int and manifest['lifetime_seconds'] == 300
            and manifest['task_context_bundle'] is manifest['context_refresh_policy']
            is manifest['published_retrieval_policy'] is None
            and type(manifest['bootstrap_listener_inode']) is int and manifest['bootstrap_listener_inode'] > 0,
            'actual bounded historical native runtime profile')
    scope = envelope['payload']
    same(manifest['owner_identity'],scope['native_population']['owner_identity'], 'historical grant exact native owner identity')
    route = scope['native_population']['execution_route_policy']
    require(manifest['execution_route_policy_id'] == route['policy_id']
            and manifest['baseline_commit'] == result['original_commit']
            and manifest['repository_root'] == '/results/native/repository',
            'historical grant native route and original source baseline')
    original_custody = scope['source_custody']
    self_cid(original_custody,'custody_cid')
    require(scope['advisory_closure']['signed_closure']['payload']['source']['custody_cid'] == original_custody['custody_cid'],
            'actual launch scope source custody is exact prepared advisory source custody')
    manifest_source = load(mapped('/results/native/cas/structured/bagu/'+result['original_head']['manifest_cid']))
    require(manifest['baseline_tree_id'] == manifest_source['snapshot']['git_tree'], 'actual runtime grant exact original Git tree')
    require(manifest['argv_digest'] == digest(manifest['argv']) and type(manifest['argv']) is list
            and all(type(item) is str for item in manifest['argv']), 'entire actual signed runtime argv identity')
    argv = manifest['argv']
    def argument(name):
        require(argv.count(name) == 1, 'one exact signed runtime option: '+name)
        index = argv.index(name)
        require(index+1 < len(argv), 'signed runtime option value required')
        return argv[index+1]
    require(argument('--implementation-command') == scope['candidate']['implementation_command']
            and argument('--worktree-root') == manifest['worker_worktree_root'] == '/opt/ipfs-supervisor/worktrees'
            and argument('--authority-mode') == 'quack'
            and argument('--task-source-kind') == 'duckdb'
            and argument('--quack-endpoint') == manifest['owner_identity']['listen_uri']
            and argument('--endpoint-secret-handle') == manifest['owner_identity']['secret_handle']
            and argument('--state-store-id') == manifest['owner_identity']['store_id']
            and argument('--state-store-generation') == str(manifest['owner_identity']['generation'])
            and argument('--state-schema-revision') == str(manifest['owner_identity']['schema_revision'])
            and argument('--state-owner-client-id') == manifest['bootstrap_client_id']
            and argument('--todo-path') == '/results/native/private/intent.duckdb',
            'actual runtime argv joins native owner route, credential reference and exact candidate command')
    selected = [argv[index+1] for index,item in enumerate(argv) if item == '--execution-slice-task-cid']
    require(len(selected) == 2 and len(set(selected)) == 2, 'actual runtime retains two distinct selected native task CIDs')
    same(sorted(selected),sorted(task['task_cid'] for task in scope['native_population']['tasks']),
         'actual signed runtime CLI retains complete original native task population')
    same(scope['native_population']['selected_task_cids'],[scope['candidate']['descriptor']['task_cid']],
         'native execution scope selects exactly the one ready residual candidate while retaining both original tasks')
    aliases = [argv[index+1] for index,item in enumerate(argv) if item == '--execution-slice-task-id']
    require(len(aliases) == len(selected), 'one explicit CLI task alias per selected native task CID')
    expected_aliases = {task['task_cid']:task['task_alias'] for task in scope['native_population']['tasks']}
    same(dict(zip(selected,aliases)),expected_aliases, 'each actual runtime CLI task CID joins its original native task alias')
    launcher = manifest['finite_worker_launcher']
    require(set(launcher) == {'path','sha256'} and launcher['path'] == '/opt/ipfs-supervisor/bin/owner-worker'
            and sha(read(mapped(launcher['path']))) == launcher['sha256']
            and scope['candidate']['argv'][0] == launcher['path'],
            'actual signed runtime exact installed owner launcher byte seal')
    environment = manifest['environment']
    require(type(environment) is list and all(type(row) is list and len(row) == 2
            and all(type(item) is str for item in row) for row in environment)
            and len({row[0] for row in environment}) == len(environment), 'closed distinct actual signed runtime environment')
    environment = dict(environment)
    expected_environment_names = {'PYTHONPATH','PYTHONUNBUFFERED','GIT_CONFIG_NOSYSTEM','GIT_CONFIG_GLOBAL',
        'GIT_CONFIG_COUNT','GIT_TERMINAL_PROMPT','GIT_AUTHOR_NAME','GIT_AUTHOR_EMAIL','GIT_COMMITTER_NAME',
        'GIT_COMMITTER_EMAIL','GIT_CONFIG_PARAMETERS','IPFS_SUPERVISOR_CANDIDATE_RUNNER_BINDING'}
    expected_environment_names.update('GIT_CONFIG_'+kind+'_'+str(index) for kind in ('KEY','VALUE') for index in range(6))
    require(set(environment) == expected_environment_names, 'closed complete native source runtime environment names')
    require(environment['PYTHONPATH'] == '/opt/ipfs-supervisor/source:/opt/ipfs-supervisor/source:/opt/ipfs-supervisor/datasets:/opt/ipfs-supervisor/kit'
            and environment['PYTHONUNBUFFERED'] == '1' and environment['GIT_CONFIG_NOSYSTEM'] == '1'
            and environment['GIT_CONFIG_GLOBAL'] == '/dev/null' and environment['GIT_TERMINAL_PROMPT'] == '0'
            and environment['GIT_CONFIG_COUNT'] == '6' and environment['GIT_CONFIG_KEY_0'] == 'core.hooksPath'
            and environment['GIT_CONFIG_VALUE_0'] == '/dev/null' and environment['GIT_CONFIG_KEY_1'] == 'core.fsmonitor'
            and environment['GIT_CONFIG_VALUE_1'] == 'false'
            and environment['GIT_CONFIG_KEY_2'] == 'safe.directory'
            and environment['GIT_CONFIG_VALUE_2'] == '/opt/ipfs-supervisor/source'
            and environment['GIT_CONFIG_KEY_3'] == 'gc.auto' and environment['GIT_CONFIG_VALUE_3'] == '0'
            and environment['GIT_CONFIG_KEY_4'] == 'gc.autoDetach' and environment['GIT_CONFIG_VALUE_4'] == 'false'
            and environment['GIT_CONFIG_KEY_5'] == 'maintenance.auto' and environment['GIT_CONFIG_VALUE_5'] == 'false'
            and environment['GIT_CONFIG_PARAMETERS'] == ''
            and environment['GIT_AUTHOR_NAME'] == environment['GIT_COMMITTER_NAME'] == 'Isolated Supervisor'
            and environment['GIT_AUTHOR_EMAIL'] == environment['GIT_COMMITTER_EMAIL'] == 'supervisor@example.invalid',
            'actual selected native source environment and Git controls')
    same(parse(environment['IPFS_SUPERVISOR_CANDIDATE_RUNNER_BINDING']),manifest['candidate_runner'],
         'actual runtime signed environment contains exact isolated candidate runner binding')
    runner = manifest['candidate_runner']
    require(set(runner) == {'schema','argv','completion_authority','files','namespaces','owner_uid','worker_uid'}
            and runner['schema'] == 'supervisor-isolated-candidate-runner@1'
            and runner['argv'] == ['/opt/ipfs-supervisor/bin/validation-worker']
            and set(runner['files']) == {'/opt/ipfs-supervisor/bin/validation-worker',
                '/opt/ipfs-supervisor/bin/worker-entry','/opt/ipfs-supervisor/container-boundary.json'},
            'closed exact selected isolated native validation runner')
    for runner_path,checksum in runner['files'].items():
        if runner_path == '/opt/ipfs-supervisor/container-boundary.json':
            actual_path = mapped(runner_path)
        else:
            actual_path = mapped(runner_path)
        require(sha(read(actual_path)) == checksum, 'actual runtime candidate runner installed bytes')
    boundary_raw = pin(load(ROOT/'container-execution-final.json')['container_boundary_artifact'])
    boundary = parse(boundary_raw)
    final = load(ROOT/'container-execution-final.json')
    observation = load(ROOT/'offline-preflight.json')['observation']
    same(boundary,{'schema':'supervisor-container-worker-boundary@1',
        'container_id':final['container_id'],'image_id':final['image_id'],
        'owner_uid':1000,'worker_uid':1001,'single_worker':True,
        'namespaces':observation['namespaces'],
        'allowed_worktree_roots':['/opt/ipfs-supervisor/worktrees'],
        'owner_private_paths':['/opt/ipfs-supervisor/state','/results/native/private'],
        'validation_repository_roots':['/results/native/repository']},
        'entire actual deployed boundary bytes and independent offline container namespaces')
    same(manifest['candidate_runner']['namespaces'],boundary['namespaces'],
         'actual runtime runner uses exact retained deployed container namespaces')
    require(manifest['candidate_runner']['owner_uid'] == boundary['owner_uid'] == 1000
            and manifest['candidate_runner']['worker_uid'] == boundary['worker_uid'] == 1001
            and manifest['candidate_runner']['completion_authority'] is False,
            'actual historical runtime isolated runner UID and authority')
    same(manifest['local_admission'],admission['local_admission'],
         'actual runtime launch carries exact original local planning parent')
    return {'path':str(path),'signature_verified':True,'exact_signed_execution_scope':True,
            'historical_native_owner_route_verified':True,'installed_launcher_bytes_verified':True,
            'installed_runner_and_boundary_bytes_verified':True,
            'current_runtime_grant_claimed':False,'process_origin_attested':False}


def advisory_native_no_effect_refusal_audit(row, scope):
    """Join failed START/empty-tree proof to both actual durable journals."""
    require('start' in row and 'start_exception' not in row, 'typed before-Popen refusal returns an actual native operation result')
    result = row['start']
    require(result['operation'] == 'start' and result['status'] == 'conflict'
            and result['effects'] == [] and result['error']['code'] == 'conflict'
            and result['error']['details']['exception_type'] == 'VerifiedBeforePopenRefusal',
            'actual typed native before-Popen conflict with zero applied effects')
    observation = result['data']['before_popen_refusal']
    require(set(observation) == {'proof','compensation_scope'}
            and observation['compensation_scope'] == 'no-op after native empty-tree observation; no signal, repair or job executed',
            'closed bounded native no-op refusal scope')
    proof = observation['proof']
    proof_fields = {'schema','boundary','request_id','operation','repository_id','tree_id','objective_id',
        'objective_revision','policy_id','policy_revision','caller','idempotency_key','lease_id','fencing_epoch',
        'target_id','transition_id','journal_revision','saga_phase','empty_tree','applied_effect_ids'}
    require(set(proof) == proof_fields and len(wire(proof)) <= 16*1024
            and proof['schema'] == 'native-before-popen-no-effect-refusal@1'
            and proof['operation'] == 'start' and proof['saga_phase'] == 'failed'
            and proof['applied_effect_ids'] == [] and type(proof['journal_revision']) is int
            and proof['journal_revision'] == 2, 'closed exact native failed START no-effect proof')
    state = NATIVE/'private/advisory-spawn-controls'/row['case']/'launch/state'
    grant = load(state/'local-process-grant.json')['manifest']
    expected_profile = {
        'schema':'ipfs_accelerate_py/agent-supervisor/lifecycle-profile@1',
        'target_id':grant['repository_id'],'run_id':grant['run_id'],'configuration_root':digest(grant),
        'repository_root':grant['repository_root'],'state_root':grant['state_root'],
        'run_root':grant['state_root']+'/run','argv':grant['argv'],'cwd':grant['repository_root'],
        'environment':sorted(grant['environment']),'health_path':grant['state_root']+'/run/admitted_supervisor_status.json',
        'health_stale_ms':5000}
    boundary = proof['boundary']
    expected_boundary = {key:expected_profile[key] for key in (
        'target_id','run_id','configuration_root','repository_root','state_root','run_root')}
    expected_boundary.update(profile_id=digest(expected_profile),lease_id=proof['lease_id'],
        fencing_epoch=proof['fencing_epoch'],finite_scope_lease_id=scope['lease']['lease_id'],
        production_activated=False,authenticated_process_origin=False,atomicity_attested=False)
    same(boundary,expected_boundary, 'exact independently derived signed runtime profile and native lease boundaries')
    for field in ('request_id','operation','repository_id','tree_id','objective_id','policy_id','caller','idempotency_key'):
        same(proof[field],result[field], 'before-Popen proof exact original operation result binding: '+field)
    require(proof['target_id'] == proof['repository_id'] == grant['repository_id']
            and proof['tree_id'] == grant['baseline_tree_id']
            and proof['objective_id'] == proof['objective_revision'] == proof['policy_id']
            == proof['policy_revision'] == digest(grant)
            and type(proof['fencing_epoch']) is int and proof['fencing_epoch'] >= 1,
            'native refusal complete runtime configuration, original tree and fencing binding')
    require(proof['caller'] == row['execution_scope']['binding']['identity'], 'original signed owner is actual no-effect request caller')
    empty = proof['empty_tree']
    require(set(empty) == {'schema','profile_id','run_id','members','captured_at_ms','tree_id'}
            and empty['schema'] == 'ipfs_accelerate_py/agent-supervisor/process-tree@1'
            and empty['members'] == [] and empty['profile_id'] == boundary['profile_id']
            and empty['run_id'] == boundary['run_id'] and type(empty['captured_at_ms']) is int
            and empty['tree_id'] == digest({key:item for key,item in empty.items() if key != 'tree_id'}),
            'whole fresh empty native Linux process tree observation')
    def journal(name):
        raw = read(state/name,8*1024**2)
        entries = [parse(line) for line in raw.splitlines() if line]
        require(0 < len(entries) <= 256 and all(type(item) is dict for item in entries),
                'bounded complete native lifecycle/control journal')
        return entries
    saga = journal('lifecycle-transitions.jsonl')
    transactions = journal('control-transactions.jsonl')
    starts = [item for item in saga if item['intent']['request_id'] == proof['request_id']]
    require([item['phase'] for item in starts] == ['prepared','starting_new','failed']
            and [item['revision'] for item in starts] == [0,1,2]
            and all(type(item['revision']) is int for item in starts), 'actual failed START terminal saga journal sequence')
    failed = starts[-1]
    intent = failed['intent']
    require(intent['schema'] == 'ipfs_accelerate_py/agent-supervisor/lifecycle-transition-intent@1'
            and intent['transition_id'] == digest({key:item for key,item in intent.items() if key != 'transition_id'}),
            'actual original native START transition complete immutable identity')
    same(intent,starts[0]['intent'],'failed native START has exact original prepared intent')
    same(intent,starts[1]['intent'],'failed native START has exact actual pre-Popen intent')
    for field in ('request_id','repository_id','tree_id','objective_id','objective_revision','policy_id',
                  'policy_revision','caller','idempotency_key','lease_id','fencing_epoch','target_id','transition_id'):
        same(proof[field],intent[field], 'native no-effect proof persisted START intent join: '+field)
    require(intent['action'] == 'start' and intent['profile_id'] == boundary['profile_id']
            and intent['created_at_ms'] <= empty['captured_at_ms']
            and intent['configuration_root'] == digest(grant)
            and failed['failure_code'] == 'refused_before_popen_without_process_effect'
            and failed['revision'] == proof['journal_revision'], 'actual persisted failure before terminal STOP')
    for field in ('profile_id','target_id','run_id','configuration_root','repository_root','state_root','run_root'):
        same(intent[field],boundary[field], 'persisted native failed saga exact runtime boundary field: '+field)
    require(all(set(item) == {'schema','intent','phase','revision','old_tree','new_tree','old_tree_fenced',
        'health_window_started_at_ms','observed_effects','compensation','failure_code','receipt'}
            and item['schema'] == 'ipfs_accelerate_py/agent-supervisor/lifecycle-transition-saga@1'
            and item['old_tree'] is item['new_tree'] is item['receipt'] is None
            and item['old_tree_fenced'] is False and item['observed_effects'] == []
            and item['compensation'] == [] for item in starts), 'all actual START saga rows have no process or compensation effect')
    mutation = [item for item in transactions if item['request_id'] == proof['request_id']]
    require([item['phase'] for item in mutation] == ['prepared','dispatching','compensated']
            and [item['revision'] for item in mutation] == [0,1,2]
            and all(type(item['revision']) is int for item in mutation)
            and all(item['applied_effect_ids'] == [] and item['recovery_action'] == 'none' for item in mutation),
            'actual control mutation durably terminal with no applied effect or repair action')
    terminal = mutation[-1]
    transaction_identity_fields = {'schema','contract_version','request_id','operation','repository_id','tree_id',
        'objective_id','objective_revision','policy_id','policy_revision','caller','idempotency_key',
        'lease_id','fencing_epoch','effect_ids'}
    transaction_fields = transaction_identity_fields | {'transaction_id','phase','revision','applied_effect_ids',
        'recovery_action','failure_code','result','updated_at_ms'}
    require(all(set(item) == transaction_fields and item['schema']
            == 'ipfs_accelerate_py/agent-supervisor/control-mutation-transaction@1'
            and type(item['contract_version']) is int and item['contract_version'] == 1
            and item['transaction_id'] == digest({key:item[key] for key in transaction_identity_fields})
            for item in mutation), 'complete actual native no-effect control immutable transaction identities')
    self_cid(terminal['result'],'content_id')
    same({key:item for key,item in terminal['result'].items() if key != 'content_id'},result,
         'actual native terminal mutation persists exact returned refusal result and independently verified record CID')
    projected = result['data']['transaction']
    for key in terminal:
        if key not in {'result','updated_at_ms'}:
            same(projected[key],terminal[key],'returned terminal mutation exact persisted row field: '+key)
    require(projected['result'] is None and terminal['failure_code'] == result['error']['code']
            and terminal['effect_ids'] == intent['expected_effect_ids'] == ['start:isolated-process-tree'],
            'actual declared START effect remains unapplied in terminal control projection')
    for field in ('request_id','repository_id','tree_id','objective_id','objective_revision','policy_id',
                  'policy_revision','caller','idempotency_key','lease_id','fencing_epoch'):
        same(terminal[field],proof[field],'terminal no-effect control transaction exact native proof binding: '+field)
    stop = row['stop']
    stop_saga = [item for item in saga if item['intent']['request_id'] == stop['request_id']]
    stop_mutation = [item for item in transactions if item['request_id'] == stop['request_id']]
    require(len(saga) == 7 and len(transactions) == 6
            and [item['phase'] for item in stop_saga] == ['prepared','stopping_old','old_fenced','committed']
            and [item['revision'] for item in stop_saga] == [3,4,5,6]
            and all(type(item['revision']) is int for item in stop_saga)
            and [item['phase'] for item in stop_mutation] == ['prepared','dispatching','committed']
            and [item['revision'] for item in stop_mutation] == [0,1,2]
            and all(type(item['revision']) is int for item in stop_mutation)
            and stop_saga[-1]['phase'] == 'committed'
            and stop_mutation[-1]['phase'] == 'committed'
            and saga.index(failed) < saga.index(stop_saga[0])
            and transactions.index(terminal) < transactions.index(stop_mutation[0])
            and stop_saga[0]['intent']['created_at_ms'] >= empty['captured_at_ms'],
            'actual committed native STOP follows persisted terminal failed START')
    self_cid(stop_mutation[-1]['result'],'content_id')
    same({key:item for key,item in stop_mutation[-1]['result'].items() if key != 'content_id'},stop,
         'actual successful STOP result persisted with independently verified native record CID')
    require(not any(item['phase'] in {'partial_failure','repair_required','compensation_required'}
                    for item in [*saga,*transactions]), 'no unresolved partial native launch or repair journal is reclassified')
    return {'native_failed_START_saga_verified':True,'native_terminal_no_effect_transaction_verified':True,
            'fresh_empty_tree_observation_verified':True,'native_STOP_after_terminal_START_verified':True,
            'repair_job_or_signal_claimed':False,'authenticated_process_origin':False}


def advisory_native_completion_retirement_audit(observation, state, stop):
    """Check the retained cooperative exact-handle retirement observation.

    The four original public service fields contain no opaque binding token.
    The receipt is historical runtime evidence, not a new signed grant or
    authenticated attestation of the owner process.
    """
    require(observation.get('service_retired') is True
            and observation.get('owner_validation_handler_unbound') is True
            and observation.get('owner_validation_binding_unbound') is True
            and type(observation.get('owner_validation_callbacks_active')) is int
            and observation['owner_validation_callbacks_active'] == 0
            and observation.get('retirement_token_serialized') is False,
            'actual native service, exact handler/binding and inactive owner callback retirement')
    require(stop['status'] == 'succeeded' and stop['data']['old_tree_fenced'] is True
            and stop['data']['isolated_worker_cleanup']['returncode'] == 0
            and stop['data']['isolated_worker_cleanup']['worker_uid'] == 1001,
            'retirement follows actual native STOP/fencing and isolated worker UID cleanup')
    receipt_path = state/'completion-service-close-receipt.json'
    raw = read(receipt_path,16*1024)
    receipt = parse(raw)
    same(receipt,{'schema':'supervisor-local-owner-validation-service@1','bound':True,
        'operation':'local.task.validation.run','completion_authority':False,
        'retired':True,'detached_exact_handler':True},
        'closed whole native completion-service retirement receipt without serialized opaque token')
    require(read(state/'receipts'/(cid(receipt)+'.json'),16*1024) == raw,
            'whole native retirement receipt equals actual content-addressed runtime receipt')
    # STOP's complete operation result is independently joined to its actual
    # control/lifecycle journals by the native worker/no-effect audit.
    same(load(state/'stop-receipt.json'),stop, 'retirement namespace retains exact successful STOP observation')
    return {'native_retirement_receipt_verified':True,'original_service_wire_fields_preserved':True,
            'exact_handler_and_binding_unbound_observed':True,'active_callbacks_observed':0,
            'retirement_token_serialized':False,'historical_unsigned_runtime_receipt':True,
            'authenticated_process_origin':False}


def advisory_closure_audit(result, admission, contexts, bridge, worker, plain, cas):
    """Audit archived launch bytes without treating copied inodes as live ones."""
    require(result['advisory_final_spawn_closure_qualified'] is True,
            'actual cooperative final-spawn qualification required')
    retirements = [advisory_native_completion_retirement_audit(
        result,NATIVE/'private/launch/state',result['stop'])]
    controls = load(NATIVE/'advisory-spawn-controls.json')
    same(controls,result['advisory_spawn_controls'],'complete native control receipts')
    require(type(controls) is list and len(controls) == 2
            and len({row['case'] for row in controls}) == 2,'two distinct native late mutations')
    snapshots = load(NATIVE/'advisory-artifact-snapshots.json')
    require(set(snapshots) == {'schema','closures'}
            and snapshots['schema'] == 'finite-advisory-launch-artifact-snapshots@1',
            'closed historical launch byte index')
    indexed = {row['closure_cid']:row for row in snapshots['closures']}
    require(len(indexed) == len(snapshots['closures']) == 3,
            'one archive per negative and positive launch seal')
    references = [row['closure_reference'] for row in controls] + [result['advisory_closure_reference']]
    same(plain['advisory_spawn_closures'],references,'all three full native seals hydrated as metadata')
    same(plain['advisory_spawn_controls'],[*controls,result['advisory_drift_during_stop']],
         'both launch refusals and actual STOP drift observation hydrated as metadata')
    same(plain['advisory_launch_artifacts'],[{**snapshots,'byte_payload_hydrated':False,
         'scope':'complete separately retained original seal byte populations indexed by native metadata'}],
         'whole original byte archive population indexed by hydrated metadata')
    signer = {key:admission['declaration']['binding'][key] for key in ('identity','profile_id')}
    authority = set(CLAIMS) | {'authenticated_process_origin','atomicity_attested','model_enabled_for_signed_worker'}
    limits = {'files':512,'directories':128,'file_bytes':64*1024**2,'total_bytes':128*1024**2,
              'material_bytes':128*1024,'inventory_entries':1024,'path_bytes':4096}
    archive_blobs = {}
    for archived_closure in snapshots['closures']:
        require(set(archived_closure) == {'closure_cid','files','seal_snapshot'}, 'closed complete launch archive record')
        for archived_row in [*archived_closure['files'],archived_closure['seal_snapshot']]:
            require(set(archived_row) == {'original','snapshot'}, 'closed original launch byte archive pair')
            snapshot = archived_row['snapshot']
            require(set(snapshot) == {'path','bytes','sha256'} and type(snapshot['bytes']) is int
                    and 0 <= snapshot['bytes'] <= limits['file_bytes']
                    and re.fullmatch('[0-9a-f]{64}',snapshot['sha256']) is not None
                    and snapshot['path'] == '/results/native/private/advisory-artifact-snapshots/'+snapshot['sha256']+'.blob',
                    'closed finite content-addressed original launch archive blob')
            require(snapshot['path'] not in archive_blobs or archive_blobs[snapshot['path']] == snapshot,
                    'deduplicated original launch archive has one exact descriptor per blob')
            archive_blobs[snapshot['path']] = snapshot
    require(sum(row['bytes'] for row in archive_blobs.values()) <= limits['total_bytes'],
            'entire physical deduplicated original launch archive within 128 MiB')
    same(sorted(str(path) for path in (NATIVE/'private/advisory-artifact-snapshots').iterdir()),
         sorted(str(mapped(path)) for path in archive_blobs),
         'all separately retained original launch blobs are exactly indexed')
    summaries = []
    descriptor = load(NATIVE/'candidate-descriptor.json')
    lowering = load(ADVISORY/'before-lowering.json')
    generated = load(ADVISORY/'generated-candidate.json')
    for reference in references:
        require(set(reference) == {'schema','profile','closure_cid','signed_closure','artifact','authority'}
                and reference['schema'] == 'supervisor-finite-advisory-artifact-closure-reference@1'
                and reference['profile'] == 'finite-repository-advisory-artifact-closure@1'
                and len(wire(reference)) <= limits['material_bytes'], 'closed bounded inert signed reference')
        artifact_pin = reference['artifact']
        require(set(artifact_pin) == {'path','size_bytes','sha256','witness'}
                and type(artifact_pin['size_bytes']) is int and 0 <= artifact_pin['size_bytes'] <= limits['material_bytes']
                and type(artifact_pin['witness']) is list and len(artifact_pin['witness']) == 9
                and all(type(item) is int and 0 <= item < 2**64 for item in artifact_pin['witness'])
                and stat.S_ISREG(artifact_pin['witness'][2]) and artifact_pin['witness'][5] == 1
                and artifact_pin['witness'][6] == artifact_pin['size_bytes']
                and re.fullmatch('[0-9a-f]{64}',artifact_pin['sha256']) is not None
                and Path(artifact_pin['path']).is_absolute() and '..' not in Path(artifact_pin['path']).parts
                and len(os.fsencode(artifact_pin['path'])) <= limits['path_bytes']
                and Path(artifact_pin['path']).name == 'closure.json',
                'closed bounded original signed closure file descriptor witness')
        require(set(reference['authority']) == authority,'exact advisory reference authority population')
        false(reference['authority'],authority)
        envelope = reference['signed_closure']
        require(reference['closure_cid'] == cid(envelope),'whole advisory seal CID')
        value = public_signature(envelope,signer)
        require(set(value) == {'schema','profile','scope','head','finite_admission_cid','semantic_context_cid',
                'candidate','administrator_task_cids','source','model','lowering','proposal','limits','files',
                'directories','absent_paths','implementation','authority'}
                and value['schema'] == 'supervisor-finite-advisory-artifact-closure@1'
                and value['profile'] == reference['profile']
                and value['scope'] == 'prepared_advisory_bytes_at_existing_native_worker_process_birth'
                and len(wire(envelope)) <= limits['material_bytes'], 'closed signed launch observation profile')
        same(value['limits'],limits,'unchanged advisory bounds')
        require(set(value['authority']) == authority,'exact advisory payload authority population')
        false(value['authority'],authority)
        same(value['head'],result['original_head'],'advisory seal original head')
        same(value['candidate'],descriptor,'complete native public candidate descriptor')
        require(value['finite_admission_cid'] == cid(admission)
                and value['semantic_context_cid'] == worker['semantic_context_cid'], 'exact original native admission')
        same(value['administrator_task_cids'],bridge['administrator_task_cids'],'complete original task population')
        same(value['model'],{'training_context':contexts['root-training'],
                            'frozen_context':contexts['root-frozen']}, 'exact distinct train/frozen receipts')
        same(value['proposal'],{**{key:bridge[key] for key in (
            'reviewed_candidate_cid','generated_result_cid','worker_candidate_cid',
            'replacement_cid','replacement_sha256')},'bridge':bridge},
            'complete independently checked source/model/proposal/bridge identity')
        lower_keys = {'result_cid','translation_cid','contract_cid','tool_policy_cid',
                      'process_policy_cid','source_cid','status','scope'}
        same(value['lowering'],{**{key:lowering[key] for key in lower_keys},
             'lean_certificate_cid':cid(lowering['lean_certificate']),
             'lean_olean_cid':lowering['artifacts']['lean_olean']['cid']},
             'complete initial mathematical lowering and compiled Lean binding')
        same({key:value['source'][key] for key in ('source_cid','git_commit','manifest_cid',
                 'ast_revision_id','semantic_state_cid')},
             {'source_cid':bridge['source_cid'],'git_commit':result['original_commit'],
              'manifest_cid':result['original_head']['manifest_cid'],
              'ast_revision_id':result['original_head']['ast_revision_id'],
              'semantic_state_cid':contexts['root-frozen']['semantic_state_cid']},
             'source, Git, native manifest, AST and semantic state identity')
        require(set(value['source']) == {'source_cid','git_commit','manifest_cid','ast_revision_id',
                 'semantic_state_cid','custody_cid'} and re.fullmatch('baguqeera[a-z2-7]+',value['source']['custody_cid']),
                'closed source binding with native preparation custody content identity')
        files = value['files']
        paths = [row['path'] for row in files]
        require(0 < len(files) <= limits['files'] and paths == sorted(set(paths))
                and sum(row['size_bytes'] for row in files) <= limits['total_bytes'], 'complete bounded seal file population')
        for row in files:
            require(set(row) == {'path','size_bytes','sha256','witness'}
                    and type(row['size_bytes']) is int and 0 <= row['size_bytes'] <= limits['file_bytes']
                    and type(row['witness']) is list and len(row['witness']) == 9
                    and all(type(item) is int and 0 <= item < 2**64 for item in row['witness'])
                    and stat.S_ISREG(row['witness'][2]) and row['witness'][5] == 1
                    and row['witness'][6] == row['size_bytes'], 'exact historical file identity witness')
            require(Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
                    and len(os.fsencode(row['path'])) <= limits['path_bytes']
                    and re.fullmatch('[0-9a-f]{64}',row['sha256']), 'closed absolute historical file path and digest')
        directories = value['directories']
        dir_paths = [row['path'] for row in directories]
        require(0 < len(directories) <= limits['directories'] and dir_paths == sorted(set(dir_paths))
                and sum(len(row['entries']) for row in directories) <= limits['inventory_entries'],
                'complete bounded directory populations')
        all_paths = set(paths) | set(dir_paths)
        for row in directories:
            require(set(row) == {'path','witness','entries'} and len(row['witness']) == 5
                    and all(type(item) is int and 0 <= item < 2**64 for item in row['witness'])
                    and stat.S_ISDIR(row['witness'][2]) and Path(row['path']).is_absolute()
                    and '..' not in Path(row['path']).parts, 'closed original directory identity witness')
            expected_entries = [{'name':Path(path).name,'kind':'directory' if path in dir_paths else 'file'}
                                for path in all_paths if str(Path(path).parent) == row['path']]
            same(row['entries'],sorted(expected_entries,key=lambda item:item['name']),
                 'every signed directory entry joins the complete signed file/directory population')
        absent = value['absent_paths']
        require(type(absent) is list and absent == sorted(set(absent)) and len(absent) <= 8
                and not (set(absent) & all_paths)
                and all(path == '/results/native/private/advisory/train.duckdb.wal' for path in absent),
                'closed originally absent registry companion population')
        implementation = value['implementation']
        require(type(implementation) is dict and 0 < len(implementation) <= 128
                and all(type(name) is str and re.fullmatch('[0-9a-f]{64}',digest)
                        and any(row['sha256'] == digest and row['path'].endswith('/'+name.replace('.','/')+'.py')
                                for row in files) for name,digest in implementation.items()),
                'all selected producer module identities join actual signed file bytes')
        required = {descriptor['artifact'],lowering['output']+'/result.json',generated['output']+'/result.json'}
        required.update(row['path'] for row in lowering['artifacts'].values())
        required.update(row['path'] for row in generated['artifacts'].values())
        for name in ('root-training','root-frozen'):
            base = '/results/native/private/advisory/'+name+'-context'
            required.add(base+'/context.json')
            required.update(base+'/'+row['relative_path'] for row in contexts[name]['artifacts'].values())
        required.add('/results/native/private/advisory/train.duckdb')
        require(required <= set(paths),'independently enumerated context, lowering, proposal, handoff and registry files')
        archive = indexed[reference['closure_cid']]
        require(set(archive) == {'closure_cid','files','seal_snapshot'},'closed original byte archive entry')
        same([row['original'] for row in archive['files']],files,'no omitted or substituted archived launch file')
        for row in archive['files']:
            raw = pin(row['snapshot'])
            require(len(raw) == row['original']['size_bytes'] and sha(raw) == row['original']['sha256'],
                    'archived bytes equal signed prelaunch original file')
        same(archive['seal_snapshot']['original'],reference['artifact'],'original retained closure file pin')
        require(pin(archive['seal_snapshot']['snapshot']) == wire(envelope)
                and pin(reference['artifact']) == wire(envelope),'retained signed seal bytes')
        independent = advisory_original_artifact_audit(value,archive,contexts,lowering,generated,descriptor,cas)
        summaries.append({'closure_cid':reference['closure_cid'],'files':len(files),
                          'bytes':sum(row['size_bytes'] for row in files),'signature_verified':True,
                          'independent_original_artifact_audit':independent})
    scope = public_signature(load(NATIVE/'execution-scope.json'),signer)
    require(scope['schema'] == 'supervisor-finite-repository-advisory-execution-scope@1'
            and scope['profile'] == 'finite-repository-one-ready-advisory-bound-native-worker@1',
            'new signed execution profile with exact advisory closure')
    same(scope['advisory_closure'],references[-1],'entire positive advisory reference signed by execution scope')
    runtime_grants = [advisory_historical_runtime_grant_audit(
        NATIVE/'private/launch/state/local-process-grant.json',
        load(NATIVE/'execution-scope.json'),admission,signer,result)]
    no_effect_refusals = []
    for row in controls:
        negative_scope = public_signature(row['execution_scope'],signer)
        require(negative_scope['schema'] == scope['schema'] and negative_scope['profile'] == scope['profile']
                and negative_scope['finite_admission_cid'] == scope['finite_admission_cid'],
                'negative runtime receives the same original native admission and advisory profile')
        same(negative_scope['advisory_closure'],row['closure_reference'],'whole negative seal in signed execution scope')
        same(negative_scope['candidate']['descriptor'],descriptor,'exact negative native public candidate')
        same(row['execution_scope_after_stop'],row['execution_scope'],'STOP preserves signed negative scope')
        require(row['armed_actual_popen_callback'] is True and row['after_verified_callback'] is True
                and type(row['children_count']) is int and row['children_count'] == 0
                and type(row['workerPopen_count']) is int and row['workerPopen_count'] == 0
                and all(type(row[key]) is int and row[key] == 0 for key in ('claim_delta','attempt_delta','publication_delta'))
                and row['remaining_processes'] == 0 and row['envelope_released'] is True,
                'actual late callback refusal with no observed process, claim, attempt or publication')
        retirements.append(advisory_native_completion_retirement_audit(
            row,NATIVE/'private/advisory-spawn-controls'/row['case']/'launch/state',row['stop']))
        require(row['task_population_unchanged'] is True and row['remaining_processes_before_stop'] == 0
                and row['bootstrap_errors'] == []
                and row['native_leases_after']['active_lease_count'] == 0
                and row['native_leases_after']['waiting_request_count'] == 0,
                'actual no-process refusal and native resource cleanup')
        same(row['task_rows_before'],row['task_rows_after'],'full original task rows unaffected by refused START')
        same(row['task_rows_before'],result['task_rows_before_candidate'],
             'negative control full task rows are exact original pre-candidate native population')
        population = negative_scope['native_population']['tasks']
        require(len(population) == 2 and len({task['task_cid'] for task in population}) == 2,
                'complete distinct signed original native control task population')
        same(row['task_rows_before'],{task['task_cid']:task for task in population},
             'actual control task rows join complete signed native scope population')
        runtime_grants.append(advisory_historical_runtime_grant_audit(
            NATIVE/'private/advisory-spawn-controls'/row['case']/'launch/state/local-process-grant.json',
            row['execution_scope'],admission,signer,result))
        same(row['native_effect_rows_before'],row['native_effect_rows_after'],
             'actual native claims, attempts, merge attempts and completion receipts unaffected')
        require(set(row['native_effect_rows_before']) == {'task_claims','task_attempts','merge_attempts','completion_receipts'}
                and row['publication_head_before'] == row['publication_head_after'] == result['original_commit'],
                'complete selected effect projections and unchanged publication head')
        no_effect_refusals.append(advisory_native_no_effect_refusal_audit(row,negative_scope))
        require(row['start_refused'] is True and (
                ('start' in row and row['start']['status'] != 'succeeded')
                or ('start_exception' in row and type(row['start_exception']['type']) is str))
                and row['stop']['status'] == 'succeeded',
                'actual refused START and successful cleanup')
        before,changed,restored = (row[key] for key in ('before','changed','restored'))
        require(before['path'] == changed['path'] == restored['path'] == row['artifact_path']
                and before['sha256'] == restored['sha256'] and before['size_bytes'] == restored['size_bytes'],
                'native fixture artifact bytes restored before a freshly prepared seal')
        require(before['witness'] != changed['witness'],'actual persistent file witness changed')
        candidates = [item for item in row['closure_reference']['signed_closure']['payload']['files']
                      if item['path'] == row['artifact_path']]
        require(len(candidates) == 1,'mutated target is a mandatory signed artifact')
        same(before,candidates[0],'native callback starts from the exact sealed artifact identity')
        expected_targets = {'append_checkpoint_byte':'/results/native/private/advisory/root-frozen-context/checkpoint.json',
                            'replace_context_inode_same_bytes':'/results/native/private/advisory/root-frozen-context/inference.json'}
        require(row['case'] in expected_targets and row['artifact_path'] == expected_targets[row['case']],
                'exact predeclared checkpoint and frozen inference native mutations')
        require(row['stop']['data']['old_tree_fenced'] is True
                and row['stop']['data']['isolated_worker_cleanup']['returncode'] == 0
                and row['stop']['data']['isolated_worker_cleanup']['worker_uid'] == 1001,
                'actual native STOP fence and delegated isolated worker cleanup')
    require(any(row['before']['sha256'] != row['changed']['sha256'] for row in controls)
            and any(row['before']['sha256'] == row['changed']['sha256']
                    and row['before']['witness'][1] != row['changed']['witness'][1] for row in controls),
            'both byte drift and same-byte inode replacement exercised natively')
    drift = result['advisory_drift_during_stop']
    require(drift['injection'] is True and drift['before']['sha256'] != drift['changed']['sha256']
            and drift['stop_succeeded'] is True and drift['remaining_processes'] == 0,
            'native STOP and cleanup succeed despite advisory receipt byte drift')
    require(drift['artifact_path'] == '/results/native/private/advisory/root-frozen-context/inference.json',
            'actual STOP drift targets exact mandatory frozen inference receipt')
    sealed_stop_targets = [item for item in references[-1]['signed_closure']['payload']['files']
                           if item['path'] == drift['artifact_path']]
    require(len(sealed_stop_targets) == 1, 'one exact positive seal STOP mutation target')
    same(drift['before'],sealed_stop_targets[0], 'STOP drift begins from exact positive launch seal file')
    require(drift['restored']['path'] == drift['before']['path']
            and drift['restored']['size_bytes'] == drift['before']['size_bytes']
            and drift['restored']['sha256'] == drift['before']['sha256']
            and drift['changed']['witness'] != drift['before']['witness']
            and result['stop']['status'] == 'succeeded',
            'actual STOP receipt succeeds and original advisory bytes are restored')
    require(result['launch_artifact_bytes_hydrated'] is False,
            'large prelaunch original byte archive remains separately indexed')
    return {'scope':'cooperative detached observation immediately before parent Popen',
            'historical_closures':summaries,'negative_controls':2,'native_stop_with_advisory_drift':True,
            'actual_historical_runtime_grants':runtime_grants,
            'negative_native_task_population_joined':True,
            'all_three_native_validation_services_retired_after_STOP':True,
            'native_validation_service_retirements':retirements,
            'native_no_effect_START_refusals':no_effect_refusals,
            'physical_launch_archive_blobs':len(archive_blobs),
            'physical_launch_archive_bytes':sum(row['bytes'] for row in archive_blobs.values()),
            'signed_model_off_worker_preserved':True,'copied_host_inodes_compared_to_original':False,
            'current_launch_grant_verified':False,'authenticated_process_origin':False,
            'atomicity_attested':False,'launch_artifact_bytes_hydrated':False}


def wrapper_audit():
    final = load(ROOT/'container-execution-final.json')
    require(final['returncode'] == 0 and type(final['returncode']) is int
            and final['native_module_launched'] is True and final['container_removed'] is True
            and final['container_results_copied'] is True and final['candidate_handoffs_copied'] is True
            and final['host_reservation_acquired'] is final['host_reservation_released'] is True
            and final['host_lease_held_after_native_exit'] is True,'actual completed container cleanup and held host reservation')
    require(final['network'] == 'none' and final['privileged'] is False
            and final['cpu_limit'] == 12 and final['memory_limit_bytes'] == 8*1024**3 and final['pids_limit'] == 512
            and final['native_total_timeout_seconds'] == 900 and final['finite_preparation_deadline_seconds'] == 90,'exact outer deployed resource profile and unchanged child preparation cap')
    lease = final['host_reservation']
    require(lease['cpu_slots'] == 12 and lease['memory_mb'] == 8192 and lease['child_process_slots'] == 12
            and lease['requires_gpu'] is False and lease['owner_pid'] > 0,'actual host reservation units and owner observation')
    resources = final['host_resources_after_cleanup']
    require(resources['active_lease_count'] == resources['waiting_request_count'] == 0
            and resources['allocated']['cpu_slots'] == resources['allocated']['memory_mb'] == 0,'actual host leases/waiters/accounting released')
    preflight = load(ROOT/'offline-preflight.json')
    require(preflight['returncode'] == 0 and preflight['status'] == 'completed'
            and preflight['container_id'] == final['container_id'] and preflight['image_id'] == final['image_id']
            and preflight['network'] == 'none' and preflight['timeout_seconds'] == 90,'actual offline preflight belongs to same container/image')
    require(sha(read(ROOT/'offline-preflight.stdout')) == preflight['stdout_sha256']
            and sha(read(ROOT/'offline-preflight.stderr')) == preflight['stderr_sha256'],'raw preflight command output pins')
    observation = preflight['observation']
    same(parse(read(ROOT/'offline-preflight.stdout')),observation,'actual preflight stdout producer observation')
    false(observation,('proof_authority','execution_authority','completion_authority'))
    require(observation['owner_uid'] == 1000 and observation['packages']['duckdb']['version'] == '1.5.5'
            and observation['packages']['numpy']['version'] == '1.26.4'
            and observation['packages']['torch']['version'].endswith('+cpu')
            and set(observation['extension_load']['extensions']) == {'ducklake','httpfs','quack'}
            and observation['extension_load']['duckdb'] == '1.5.5'
            and observation['lean']['version'].startswith('Lean (version 4.34.1,'),'actual observed selected native packages/extension cache/Lean version')
    return {'image_id':final['image_id'],'container_id':final['container_id'],'network':'none',
            'cpu_slots':12,'memory_mb':8192,'pids_limit':512,'container_removed':True,
            'native_total_timeout_seconds':900,'finite_preparation_deadline_seconds':90,
            'package_observations':{name:value['version'] for name,value in observation['packages'].items()},
            'package_and_elf_observations_are_complete_transitive_attestation':False,
            'lean_executable':observation['lean']['executable'],
            'python_executable':observation['executable']}


def pin_manifest(arguments):
    path = arguments.manifest.absolute()
    raw = read(path,64*1024**2,external=True)
    require(re.fullmatch('[0-9a-f]{64}',arguments.manifest_sha256) is not None
            and sha(raw) == arguments.manifest_sha256,'exact externally pinned audit manifest')
    body = parse(raw)
    pins = body if type(body) is list else body['pins']
    require(type(pins) is list and pins and len({item['path'] for item in pins}) == len(pins),'complete distinct external pin population')
    if arguments.pin_count is not None:
        require(type(arguments.pin_count) is int and len(pins) == arguments.pin_count,'external expected pin count')
    total = sum(item.get('bytes',item.get('size_bytes')) for item in pins)
    if arguments.pin_bytes is not None:
        require(total == arguments.pin_bytes,'external expected total pinned bytes')
    for item in pins:
        pin(item)
    return raw,pins,total


def audit(arguments):
    global ROOT,NATIVE,ADVISORY
    ROOT = arguments.output.absolute()
    require(ROOT.is_dir() and ROOT.resolve(strict=True) == ROOT,'canonical complete qualification namespace')
    NATIVE,ADVISORY = ROOT/'native',ROOT/'native/private/advisory'
    manifest_raw,pins,pin_bytes = pin_manifest(arguments)
    result = load(NATIVE/'result.json')
    require(result['schema'] == 'finite-advisory-spawn-native-worker-qualification@1'
            and result['status'] == 'completed','actual completed joined result required; failed attempts remain distinct')
    false(result,('production_activated','learned_features_authorize_execution',
            'convergence_proved','generalization_verified','formal_decoder_available','parser_correctness_proved',
            'public_terminal_bench_task_satisfied','universal_python_semantics_proved','task_omission_authority'))
    require(result['signed_worker_feature_mode'] == 'model_off' and result['training_during_worker_execution'] == 0
            and result['provider_calls'] == 0 and result['active_leases'] == result['waiting_requests'] == 0
            and type(result['actual_attempted_training_epochs']) is int and result['actual_attempted_training_epochs'] == 32,
            'separate advisory preparation, exact training cost, no provider or retained reservations')
    for name in ('complete_task_population_retained','actual_public_checks_passed','stale_current_admission_rejected',
                 'historical_artifacts_unchanged','successor_cold_finite_outcomes_agree','no_work_successor_grants_no_task_omission',
                 'native_worker_successor_loop_qualified','verified_proposal_consumed_by_native_worker'):
        require(result[name] is True,'actual completed native fixture observation: '+name)
    helper = load(ADVISORY/'result.json')
    same(helper,result['advisory_metadata_and_cold_replay'],'entire returned advisory support result')
    require(helper['status'] == 'completed' and helper['actual_attempted_training_epochs'] == 32
            and helper['active_leases'] == helper['waiting_requests'] == 0 and helper['historical_parent_preserved'] is True,'actual completed native advisory stages')
    false(helper,CLAIMS)
    false(helper['claims'],CLAIMS)
    criteria = load(ADVISORY/'criteria.json')
    same(criteria['configuration'],{'epochs':16,'learning_rate':.01,'seed':1729},'predeclared bounded trainer configuration')
    same(criteria['selections'],[['calc.py','train'],['known_variant.py','train'],['tune.py','tune'],['canary.py','canary']],'predeclared transductive training/tuning/canary roles')
    false(criteria['claims'],CLAIMS)
    source_pins = result['execution_sources']
    copies = sorted((NATIVE/'selected-source-snapshot').glob('*.py'))
    require(len(copies) == len(source_pins) and len({item['path'] for item in source_pins}) == len(source_pins),'complete distinct selected source snapshots')
    require(sorted(sha(read(file)) for file in copies) == sorted(sha(pin(item)) for item in source_pins),'actual deployed selected sources equal retained source copies')
    # Deliberately do not compare live workspace files to historical deployment.
    # Concurrent producers may differ; only this namespace's actual generation
    # is qualified by this historical byte audit.
    wrapper = wrapper_audit()
    plain,metadata,metadata_summary = metadata_audit(helper)
    heads,cas,source_maps = source_audit(result,metadata,plain)
    contexts,training = training_audit(helper,heads)
    lowerings = lowering_audit(heads,cas,source_maps)
    admissions,declarations,receipts = admission_audit(heads)
    bridge,worker,generated,candidate = candidate_audit(admissions[0],contexts,heads)
    require(candidate['python_sha256'] == wrapper['python_executable']['sha256'],
            'signed proposal Python seal equals actual offline preflight observation')
    for checkpoint in load(ADVISORY/'phase-costs.json')['training_cost']['checkpoints']:
        require(checkpoint['worker_receipt']['executable_sha256'] == wrapper['python_executable']['sha256'],
                'both actual trainer child Python seals equal deployed preflight observation')
    native = native_worker_audit(result,admissions[0],worker,source_maps,cas)
    spawn_closure = advisory_closure_audit(result,admissions[0],contexts,bridge,worker,plain,cas)
    controls = helper['controls']
    require({item['control'] for item in controls} == {'root_context_after_publication','root_context_under_successor','root_lowering_under_successor'}
            and len(controls) == 3 and all(item['rejected'] is True for item in controls),'all three stale-root context/proof controls')
    cold = load(ADVISORY/'cold-response.json')
    same(cold,helper['cold_verification'],'exact separately retained fresh-process cold verification')
    same(cold['head'],heads[1],'cold actual durable successor native head')
    same(cold['registry_before'],cold['registry_after'],'fresh-process cold no-fit/no-promotion registry inventory')
    require(load(ADVISORY/'cold-process.json')['returncode'] == 0 and cold['status'] == 'completed'
            and cold['training_steps'] == cold['active_leases'] == cold['waiting_requests'] == 0
            and cold['same_durable_catalog_reopened'] is True
            and cold['independent_cold_catalog_rebinding_claimed'] is False
            and cold['current_facts_count'] == 2 and cold['selected_task_ids'] == [],'actual exact-head no-fit cold process')
    false(cold,CLAIMS)
    artifact_count,artifact_bytes = 0,0
    for artifact in plain['artifacts']:
        if artifact.get('schema') == 'ipfs-datasets.software-contracts.semantic-artifact@1':
            continue
        require(artifact['schema'] == 'terminal-codebase-adaptation-retained-artifact@1'
                and artifact['encoding'] == 'base64_exact_observed_bytes','complete producer byte artifact encoding')
        raw = base64.b64decode(artifact['content_base64'],validate=True)
        require(raw == pin(artifact['artifact']),'metadata whole physical artifact bytes')
        artifact_count += 1
        artifact_bytes += len(raw)
    same(plain['worker_bridge'][0],bridge,'whole source/model/candidate bridge retained in metadata')
    same(plain['lowering_proofs'],[load(ADVISORY/'before-lowering.json'),load(ADVISORY/'successor-lowering.json')],'both complete formal lowering producers in metadata')
    require(all(value['tool_policy']['lean']['sha256'] == wrapper['lean_executable']['sha256']
                for value in (load(ADVISORY/'before-lowering.json'),load(ADVISORY/'successor-lowering.json'))),'historical Lean tool seal agrees with deployed preflight observation')
    again,pins_again,total_again = pin_manifest(arguments)
    require(again == manifest_raw and total_again == pin_bytes,'external whole namespace pins unchanged after audit')
    same(pins_again,pins,'complete manifest unchanged after read-only SQL/Git/Parquet audit')
    return {'schema':VERSION,'status':'passed','observed_at':datetime.now(timezone.utc).isoformat(),
        'output':str(ROOT),'script_sha256':sha(Path(__file__).read_bytes()),
        'manifest':{'path':str(arguments.manifest.absolute()),'sha256':sha(manifest_raw),'pin_count':len(pins),'pin_bytes':pin_bytes},
        'source_snapshot_count':len(copies),'historical_selected_sources_verified':True,'live_workspace_equivalence_claimed':False,
        'signatures_verified':len(SIGNATURES),'public_signatures':SIGNATURES,'read_only_databases':DATABASES,
        'wrapper':wrapper,'training':training,'formal_lowering':lowerings,'candidate':candidate,
        'native_worker':native,'advisory_spawn':spawn_closure,'metadata':metadata_summary,'complete_artifact_count':artifact_count,
        'complete_artifact_bytes':artifact_bytes,'stale_controls_verified':3,'cold_training_steps':0,
        'signed_worker_feature_mode':'model_off','learned_features_authorize_execution':False,
        'advisory_final_spawn_closure_qualified':True,'current_execution_grant_verified':False,
        'process_origin_attested':False,'convergence_proved':False,'generalization_verified':False,
        'universal_python_semantics_proved':False,'production_activated':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--pin-count',type=int)
    parser.add_argument('--pin-bytes',type=int)
    arguments = parser.parse_args()
    try:
        report = audit(arguments)
    except Exception as error:
        print(json.dumps({'schema':VERSION,'status':'failed','error_type':type(error).__name__,'error':str(error),
                          'script_sha256':sha(Path(__file__).read_bytes()),'native_jobs_executed':0},sort_keys=True))
        raise
    print(json.dumps(report,sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    main()
