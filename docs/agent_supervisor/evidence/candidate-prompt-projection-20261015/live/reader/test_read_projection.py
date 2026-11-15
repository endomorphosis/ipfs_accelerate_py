"""Authored metadata only: no live prompts, runtime imports, provider calls or files."""
import copy
import json
import unittest
import read_projection as r


def fixture(smaller=True):
    v=dict(schema='supervisor-candidate-prompt-projection@1',view_schema='supervisor-candidate-provider-view@1',
        source_schema='supervisor-owner-candidate-router-input@3',representation_lossless=True,
        evidence_bodies_preserved=True,unknown_fields_preserved=True,**{k:False for k in r.FLAGS})
    sizes=dict(original_input=2000,native_prompt=1200,native_context=1100,native_suffix=100,
        public_instruction_block=90,candidate_view=1500 if smaller else 2200)
    for i,n in enumerate(r.VIEW_NAMES):v[n+'_bytes']=sizes[n];v[n+'_sha256']=str(i)*64
    s=dict(schema='supervisor-candidate-prompt-selection@1',requested_mode=r.MODE,
        selected_mode=r.MODE if smaller else 'legacy',fallback_reason=None if smaller else 'complete_input_not_smaller',
        selection_basis='complete_utf8_bytes',view_receipt=v,view_receipt_sha256=r.digest(r.canonical(v)),
        **{k:False for k in r.FLAGS})
    for n,size,d in (('native_complete',2400,'a'),('candidate_complete',sizes['candidate_view']+400,'b')):
        s[n+'_bytes']=size;s[n+'_sha256']=d*64
    chosen='candidate_complete' if smaller else 'native_complete'
    for k in r.PAIR_SUFFIXES:s['selected_complete'+k]=s[chosen+k]
    out=dict(schema='router-implementation-invocation@1',purpose='coding',invocation_id='authored-invocation',
        candidate_round_dispatch={},candidate_prompt_projection=s)
    for source,target in (('original_input','prompt'),('native_prompt','native_prompt')):
        for k in r.PAIR_SUFFIXES:out[target+k]=v[source+k]
    for k in r.PAIR_SUFFIXES:out['model_prompt'+k]=s['selected_complete'+k]
    out['candidate_round_dispatch']={'schema':'supervisor-candidate-dispatch-observation@2'}
    for source,target in (('provider_input','prompt'),('native_prompt','native_prompt')):
        for k in r.PAIR_SUFFIXES:out['candidate_round_dispatch'][source+k]=out[target+k]
    return out


class ReaderTests(unittest.TestCase):
    def test_selected(self):
        out=r.project_receipt(fixture())
        self.assertEqual(out['complete_input_bytes_reduced'],500)
        self.assertEqual(out['selected_mode'],r.MODE)
        self.assertFalse(out['token_measurement'])
        self.assertFalse(out['independent_prompt_replay'])
        self.assertTrue(all(out[k] is False for k in r.FLAGS))
        self.assertNotIn('view_receipt',out)

    def test_fallback(self):
        out=r.project_receipt(fixture(False))
        self.assertEqual(out['complete_input_bytes_reduced'],0)
        self.assertEqual(out['selected_mode'],'legacy')

    def test_tie_falls_back(self):
        f=fixture(False);s=f['candidate_prompt_projection'];v=s['view_receipt']
        v['candidate_view_bytes']=v['original_input_bytes'];s['candidate_complete_bytes']=s['native_complete_bytes']
        s['view_receipt_sha256']=r.digest(r.canonical(v))
        self.assertEqual(r.project_receipt(f)['selected_mode'],'legacy')

    def test_absent(self):
        f=fixture();del f['candidate_prompt_projection']
        self.assertEqual(r.project_receipt(f),dict(status='absent',independent_prompt_replay=False))

    def test_duplicate(self):
        f=fixture();out=r.project_report(dict(provider_invocations=[f,copy.deepcopy(f)]))
        self.assertEqual(out['coding_receipts'],1);self.assertEqual(out['duplicate_receipts_ignored'],1)

    def test_conflicting_duplicate(self):
        f=fixture();g=copy.deepcopy(f);g['status']='changed'
        with self.assertRaises(ValueError):r.project_report(dict(provider_invocations=[f,g]))

    def test_closed_output_ignores_dynamic_content(self):
        f=fixture();f['untrusted_provider_text']='NEVER_EXPORT_AUTHORED_SENTINEL'
        output=json.dumps(r.project_report(dict(provider_invocations=[f])))
        self.assertNotIn('NEVER_EXPORT',output);self.assertNotIn('authored-invocation',output)

    def test_empty_report(self):
        self.assertEqual(r.project_report(dict(provider_invocations=[]))['coding_receipts'],0)

    def test_non_coding_ignored(self):
        f=fixture();f['purpose']='planning'
        self.assertEqual(r.project_report(dict(provider_invocations=[f]))['coding_receipts'],0)


def mutation(name,change):
    def check(self):
        f=fixture();change(f)
        with self.assertRaises((ValueError,TypeError,KeyError)):r.project_receipt(f)
    check.__name__='test_reject_'+name
    setattr(ReaderTests,check.__name__,check)


for flag in r.FLAGS:
    mutation('selection_'+flag,lambda f,k=flag:f['candidate_prompt_projection'].__setitem__(k,True))
    mutation('view_'+flag,lambda f,k=flag:f['candidate_prompt_projection']['view_receipt'].__setitem__(k,True))
for flag in ('representation_lossless','evidence_bodies_preserved','unknown_fields_preserved'):
    mutation(flag,lambda f,k=flag:f['candidate_prompt_projection']['view_receipt'].__setitem__(k,False))
for key,val in (('schema','other'),('requested_mode','other'),('selected_mode','legacy'),
    ('fallback_reason','complete_input_not_smaller'),('selection_basis','estimated_tokens'),
    ('view_receipt_sha256','0'*64),('candidate_complete_bytes',2000),('selected_complete_bytes',1),
    ('native_complete_bytes',True),('native_complete_bytes',256001),('selected_complete_sha256','c'*64),
    ('native_complete_sha256','Z'*64),('unexpected','raw source')):
    mutation(key,lambda f,k=key,v=val:f['candidate_prompt_projection'].__setitem__(k,v))
for key,val in (('source_schema','supervisor-owner-candidate-router-input@4'),('view_schema','other'),
    ('native_context_bytes',0),('original_input_bytes',float('nan')),('unexpected','raw source')):
    mutation('view_'+key,lambda f,k=key,v=val:f['candidate_prompt_projection']['view_receipt'].__setitem__(k,v))
for key,val in (('model_prompt_bytes',0),('model_prompt_sha256','0'*64),('prompt_bytes',1),
    ('prompt_sha256','a'*64),('native_prompt_bytes',1),('native_prompt_sha256','a'*64),
    ('candidate_round_dispatch',None),('purpose','planning')):
    mutation('outer_'+key,lambda f,k=key,v=val:f.__setitem__(k,v))

mutation('dispatch_prompt_join',lambda f:f['candidate_round_dispatch'].__setitem__('provider_input_sha256','a'*64))
mutation('dispatch_schema',lambda f:f['candidate_round_dispatch'].__setitem__('schema','other'))

if __name__=='__main__':unittest.main()
