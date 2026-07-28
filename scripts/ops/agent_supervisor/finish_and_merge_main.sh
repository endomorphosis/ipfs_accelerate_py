#!/usr/bin/env bash
# Finish-and-merge helper for ASI/GOOSE integrate branch → origin/main.
#
# Preconditions:
# - ASI/GOOSE boards are completed (or you accept remaining open work)
# - Working tree is clean enough to merge
# - You have push rights to origin
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

INTEGRATE_BRANCH="${INTEGRATE_BRANCH:-integrate/finish-and-main}"
MAIN_BRANCH="${MAIN_BRANCH:-main}"
REMOTE="${REMOTE:-origin}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BOARD_CHECK="${SKIP_BOARD_CHECK:-0}"

echo "[finish] repo=$REPO_ROOT"
echo "[finish] integrate=$INTEGRATE_BRANCH main=$MAIN_BRANCH remote=$REMOTE dry_run=$DRY_RUN"

if [[ "$SKIP_BOARD_CHECK" != "1" ]]; then
  open_asi=$(python - <<'PY'
from pathlib import Path
import re
path=Path('docs/architecture/agent_supervisor_self_improvement.todo.md')
if not path.exists():
    print(0); raise SystemExit
text=path.read_text(encoding='utf-8')
heads=list(re.finditer(r'^## ASI-\d+', text, re.M))
n=0
for i,m in enumerate(heads):
    block=text[m.start(): heads[i+1].start() if i+1<len(heads) else len(text)]
    st=re.search(r'^- Status:\s*(\S+)', block, re.M)
    if not st or st.group(1).lower()!='completed':
        n+=1
print(n)
PY
)
  open_goose=$(python - <<'PY'
from pathlib import Path
import re
path=Path('docs/architecture/goose_cli_integration.todo.md')
if not path.exists():
    print(0); raise SystemExit
text=path.read_text(encoding='utf-8')
heads=list(re.finditer(r'^## GOOSE-\d+', text, re.M))
n=0
for i,m in enumerate(heads):
    block=text[m.start(): heads[i+1].start() if i+1<len(heads) else len(text)]
    st=re.search(r'^- Status:\s*(\S+)', block, re.M)
    if not st or st.group(1).lower()!='completed':
        n+=1
print(n)
PY
)
  echo "[finish] open ASI tasks=$open_asi open GOOSE tasks=$open_goose"
  if [[ "$open_asi" != "0" || "$open_goose" != "0" ]]; then
    echo "[finish] ERROR: boards still have open tasks; set SKIP_BOARD_CHECK=1 to override" >&2
    exit 2
  fi
fi

git fetch "$REMOTE" "$MAIN_BRANCH" "$INTEGRATE_BRANCH" || git fetch "$REMOTE" "$MAIN_BRANCH"

current=$(git branch --show-current)
echo "[finish] current branch=$current"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[finish] DRY RUN: would checkout $INTEGRATE_BRANCH, merge $REMOTE/$MAIN_BRANCH, push, then merge into $MAIN_BRANCH"
  git log --oneline "$REMOTE/$MAIN_BRANCH".."$INTEGRATE_BRANCH" 2>/dev/null | head -20 || \
    git log --oneline "$REMOTE/$MAIN_BRANCH"..HEAD | head -20
  exit 0
fi

# Ensure we are on integrate branch with latest local commits
git checkout "$INTEGRATE_BRANCH"
# Rebase/merge main if remote moved (prefer merge for safety with shared history)
if git rev-parse --verify "$REMOTE/$MAIN_BRANCH" >/dev/null 2>&1; then
  if ! git merge-base --is-ancestor "$REMOTE/$MAIN_BRANCH" HEAD; then
    echo "[finish] merging $REMOTE/$MAIN_BRANCH into $INTEGRATE_BRANCH"
    git merge --no-ff "$REMOTE/$MAIN_BRANCH" -m "Merge $REMOTE/$MAIN_BRANCH into $INTEGRATE_BRANCH before cutover"
  fi
fi

echo "[finish] pushing $INTEGRATE_BRANCH"
git push -u "$REMOTE" "$INTEGRATE_BRANCH"

echo "[finish] checking out $MAIN_BRANCH and merging $INTEGRATE_BRANCH"
git checkout "$MAIN_BRANCH"
git pull --ff-only "$REMOTE" "$MAIN_BRANCH" || true
git merge --no-ff "$INTEGRATE_BRANCH" -m "Merge $INTEGRATE_BRANCH: ASI/GOOSE boards complete"

echo "[finish] pushing $MAIN_BRANCH to $REMOTE"
git push "$REMOTE" "$MAIN_BRANCH"

echo "[finish] done. HEAD=$(git rev-parse --short HEAD)"
git status -sb | head -20
echo "[finish] tip:"
git log -3 --oneline
