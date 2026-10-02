"""Fresh behavioral fences around unchanged allocated-worktree materialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .repository_behavioral_admission import verify_behavioral_repository_handoff
from .repository_finite_public_context import materialize_with_public_context


def materialize_behavioral_candidate(*, repository_admission, repository_admission_sha256, **options):
    arguments = dict(artifact=repository_admission, expected_sha256=repository_admission_sha256,
        task_cid=options['task_cid'],owner_did=options['owner_did'],profile_id=options['profile_id'])
    before = verify_behavioral_repository_handoff(**arguments)
    if (before['legacy_handoff'] != dict(artifact=str(options['artifact']),sha256=options['expected_sha256'])
            or before['public_context']['artifact'] != str(options['public_context'])
            or before['public_context']['sha256'] != options['public_context_sha256']):
        raise ValueError('native worker inputs differ from the signed behavioral fence')
    result = materialize_with_public_context(**options)
    after = verify_behavioral_repository_handoff(**arguments)
    if (before['artifact_cid'],before['source_head']) != (after['artifact_cid'],after['source_head']):
        raise ValueError('repository evidence changed during allocated candidate materialization')
    result['repository_evidence_fence'] = dict(before=before,after=after,
        source_observed_before_and_after=True,atomic_filesystem_transaction=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repository-admission','repository-admission-sha256','artifact','sha256','task-cid',
                 'owner-did','profile-id','public-context','public-context-sha256'):
        parser.add_argument('--'+name,required=True)
    args=vars(parser.parse_args());args['expected_sha256']=args.pop('sha256')
    try:
        result=materialize_behavioral_candidate(**args,prompt=sys.stdin.buffer.read(256001).decode(),workspace=Path.cwd())
    except Exception as error:
        print(json.dumps(dict(status='refused',error_type=type(error).__name__,completion_authority=False)),file=sys.stderr)
        return 1
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':
    raise SystemExit(main())
