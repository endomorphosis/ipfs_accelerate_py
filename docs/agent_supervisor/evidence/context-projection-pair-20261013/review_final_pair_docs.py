"""Read-only review of retained final metadata comparison evidence and docs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, re
R=Path(__file__).resolve().parent
P=Path('/home/barberb/lift_coding/.worktrees/supervisor-comparison-contracts-20261011')
B=P/'docs/agent_supervisor/evidence/context-projection-pair-20261013'
def sha(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
recompute=json.loads((R/'context-projection-final-recomputation.json').read_text())
summary=json.loads((R/'native-observations/pair-summary.json').read_text())
audit=json.loads((R/'native-observations/exact-control-audit.json').read_text())
for label,arm in [('legacy','codex-legacy-01'),('projected','codex-projected-01')]:
 independent=recompute['arms'][label];reported=summary['arms'][arm]
 assert independent['totals']==reported['tokens']
 assert independent['receipt_sha256']==reported['evidence']['receipt']['sha256']
 assert independent['reward']['reward']==reported['official_reward']
 assert independent['source384']['coverage']==reported['source384']['coverage']
 assert independent['source384']['checkpoint_sha256']==reported['source384']['context']['checkpoint_sha256']
 assert reported['intent_ir']['status']=='fail_open_no_checkpoint'
 assert reported['source_reader']['counts']['accepted_requests']==0
 assert reported['declared_output_observation']['status']=='unavailable'
 assert reported['postimage_byte_identity_established'] is False
 assert reported['billing_total_verified'] is False
 assert reported['cache_included_in_input'] is True
 assert all(reported['durations_seconds'][k]==v for k,v in independent['durations_seconds'].items())
assert recompute['config_changed_paths']==[v['path'] for v in audit['exact_config_differences']]
assert audit['only_intentional_metadata_and_output_location_differences'] is True
copies=json.loads((B/'copy-list.json').read_text())
for row in copies['copied_files']:
 src=R/row['source_relative'];dst=B/row['evidence_relative']
 assert src.is_file() and dst.is_file() and not dst.is_symlink()
 assert src.stat().st_size==dst.stat().st_size==row['bytes']
 assert sha(src)==sha(dst)==row['sha256']
docs=[P/'docs/agent_supervisor'/name for name in ['context_projection_pair.md','reversible_codebase_views_plan.md','source_view_read_edit_pair.md','terminal_token_metadata_plan.md']]+[B/'README.md']
spec=importlib.util.spec_from_file_location('docs_links',P/'scripts/docs/check_current_docs_links.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
link_errors={str(p.relative_to(P)):module.check_file(p,check_anchors=True) for p in docs}
link_errors={k:v for k,v in link_errors.items() if v}
known_pending=['manifest.json','documentation/result.json']
assert all(k.endswith('/README.md') and all(any(target in error for target in known_pending) for error in errors) for k,errors in link_errors.items())
patterns={'private_key':r'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----','credential_token':r'(?:sk-proj-|xai-|hf_)[A-Za-z0-9_-]{20,}','jwt':r'eyJ[A-Za-z0-9_-]{12,}\.eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}','raw_prompt_field':r'"prompt"\s*:\s*\{\s*"text"\s*:'}
findings=[];inventory=[]
for path in sorted(B.rglob('*')):
 if path.is_file():
  assert not path.is_symlink()
  raw=path.read_bytes();text=raw.decode('utf-8')
  inventory.append({'path':str(path.relative_to(B)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
  for name,pattern in patterns.items():
   if re.search(pattern,text):findings.append({'file':str(path.relative_to(B)),'pattern':name})
assert findings==[],findings
regression=json.loads((R/'regressions-01/qualification.json').read_text())
assert all(sha(P/name)==digest for name,digest in regression['source_after'].items())
result={
 'schema':'context-projection-final-independent-review@1',
 'observed_at_utc':datetime.now(timezone.utc).isoformat(),
 'status':'reviewed_no_blocking_findings',
 'reviewer':'source_context_wiring',
 'review_scope':'Independent final documentation, original retained numeric receipt recomputation, matched control and public evidence review. Comparator implementation was authored by this reviewer; independent implementation review is retained separately.',
 'script':bind(Path(__file__)),
 'documents':[bind(p) for p in docs],
 'numeric_recomputation':bind(R/'context-projection-final-recomputation.json'),
 'numeric_recomputation_script':bind(R/'recompute_final_pair_review.py'),
 'pair_summary':bind(R/'native-observations/pair-summary.json'),
 'control_audit':bind(R/'native-observations/exact-control-audit.json'),
 'copy_list':bind(B/'copy-list.json'),
 'copied_artifacts_verified':len(copies['copied_files']),
 'public_inventory_count_at_review':len(inventory),
 'public_inventory_sha256':hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 'public_evidence_all_utf8_and_no_symlinks':True,
 'secret_and_raw_prompt_pattern_findings':findings,
 'doc_link_check_errors':link_errors,
 'assembly_followups':[{'path':name,'owner':'root','reason':'Final publication artifact not yet written at this review boundary; anticipated README link. Must be present and checked before publication.'} for name in known_pending if not (B/name).exists()],
 'config_differences':recompute['config_changed_paths'],
 'runtime_and_source_integrity':{'independent_recomputation_confirmed_pinned_archive_and_host_sources':True,'regression_source_closure_still_unchanged':True},
 'verified_native_metrics':{label:{'tokens':v['totals'],'official_reward':v['reward']['reward'],'provider_invocations':len(v['invocations']),'intent_ir_status':v['intent_status'],'source384_candidate_count':v['source384']['candidate_count'],'source384_verified_candidates':0,'source_broker_requests':v['indexed_reader_requests']} for label,v in recompute['arms'].items()},
 'projected_minus_legacy':recompute['projected_minus_legacy'],
 'accounting':'Final native cumulative usage counted once for each unique planning/coding invocation. Cached input is already included; reasoning output is not added again.',
 'conclusions':[
  'Native treatment used one more total token; no total-token reduction was observed.',
  'The official rewards differ 0 versus 1; one independently planned pair does not establish a causal success advantage.',
  'Dollar billing is not independently verified.',
  '882-byte selected-input reduction is within the native treatment; 885-byte offline reduction is a separate authored shared-workload comparison.',
  'Both native trials consumed the pinned Source384 checkpoint but all four decoded candidates remained unsupported and unverified; IntentIR lacked a checkpoint.',
  'No native model requested the source broker; no indexed-reader token advantage or source proof coverage is established.',
  'Final declared-output hashes are unavailable; cross-arm output byte equality is not established.',
  'The record covers one task and a separate failed Grok setup; it is not a full-suite result or a new native-Codex-harness baseline comparison.'
 ],
 'billing_advantage_claim_present':False,'causal_advantage_claim_present':False,'proof_success_claim_present':False,
 'hidden_test_or_solution_bodies_read':False,'new_provider_calls':0,'repository_files_changed_by_review':False,'blocking_findings':[]
}
(R/'context-projection-final-review.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':result['status'],'copies':result['copied_artifacts_verified'],'inventory':len(inventory),'link_errors':link_errors,'assembly_followups':result['assembly_followups'],'projected_minus_legacy':result['projected_minus_legacy']}))
