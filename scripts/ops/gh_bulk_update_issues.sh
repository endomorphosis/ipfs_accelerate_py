#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  gh_bulk_update_issues.sh --repo OWNER/REPO --start N --end M [options]

Required:
  --repo OWNER/REPO        GitHub repository (e.g. endomorphosis/ipfs_datasets_py)
  --start N                Start issue number (inclusive)
  --end M                  End issue number (inclusive)

Options:
  --milestone ID           Milestone id to set (default: 1)
  --assignee LOGIN         Assignee login to add (default: endomorphosis)
  --timeout-sec S          Per-request timeout seconds (default: 30)
  --max-retries N          Retries per issue on failure (default: 3)
  --retry-delay-sec S      Base retry delay seconds (default: 2)
  --dry-run                Print planned updates only
  --help                   Show this help

Environment overrides:
  GH_BULK_TIMEOUT_SEC
  GH_BULK_MAX_RETRIES
  GH_BULK_RETRY_DELAY_SEC
EOF
}

REPO=""
START=""
END=""
MILESTONE="1"
ASSIGNEE="endomorphosis"
TIMEOUT_SEC="${GH_BULK_TIMEOUT_SEC:-30}"
MAX_RETRIES="${GH_BULK_MAX_RETRIES:-3}"
RETRY_DELAY_SEC="${GH_BULK_RETRY_DELAY_SEC:-2}"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="${2:-}"; shift 2 ;;
    --start)
      START="${2:-}"; shift 2 ;;
    --end)
      END="${2:-}"; shift 2 ;;
    --milestone)
      MILESTONE="${2:-}"; shift 2 ;;
    --assignee)
      ASSIGNEE="${2:-}"; shift 2 ;;
    --timeout-sec)
      TIMEOUT_SEC="${2:-}"; shift 2 ;;
    --max-retries)
      MAX_RETRIES="${2:-}"; shift 2 ;;
    --retry-delay-sec)
      RETRY_DELAY_SEC="${2:-}"; shift 2 ;;
    --dry-run)
      DRY_RUN="1"; shift ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$REPO" || -z "$START" || -z "$END" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 2
fi

if ! [[ "$START" =~ ^[0-9]+$ && "$END" =~ ^[0-9]+$ ]]; then
  echo "--start and --end must be integers." >&2
  exit 2
fi

if (( START > END )); then
  echo "--start must be <= --end." >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found in PATH." >&2
  exit 127
fi

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_ISSUES=()

run_patch_once() {
  local issue_num="$1"
  local err_file="$2"
  timeout "${TIMEOUT_SEC}s" gh api -X PATCH "repos/${REPO}/issues/${issue_num}" \
    -f "milestone=${MILESTONE}" \
    -f "assignees[]=${ASSIGNEE}" \
    >/dev/null 2>"${err_file}"
}

for issue in $(seq "$START" "$END"); do
  echo "-- issue ${issue} --"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN repo=${REPO} issue=${issue} milestone=${MILESTONE} assignee=${ASSIGNEE}"
    continue
  fi

  attempt=1
  delay="$RETRY_DELAY_SEC"
  ok=0
  err_file="$(mktemp)"

  while (( attempt <= MAX_RETRIES )); do
    if run_patch_once "$issue" "$err_file"; then
      echo "UPDATED_ISSUE=${issue} attempt=${attempt}"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      ok=1
      break
    fi

    exit_code=$?
    err_text="$(tr '\n' ' ' <"$err_file" | sed 's/[[:space:]]\+/ /g' | sed 's/^ //; s/ $//')"
    echo "WARN issue=${issue} attempt=${attempt}/${MAX_RETRIES} exit=${exit_code} msg=${err_text:-<none>}" >&2

    if (( attempt < MAX_RETRIES )); then
      sleep "$delay"
      delay=$((delay * 2))
    fi
    attempt=$((attempt + 1))
  done

  rm -f "$err_file"

  if (( ok == 0 )); then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAILED_ISSUES+=("$issue")
  fi
done

echo "SUMMARY repo=${REPO} range=${START}-${END} updated=${SUCCESS_COUNT} failed=${FAIL_COUNT}"

if (( FAIL_COUNT > 0 )); then
  echo "FAILED_ISSUES=${FAILED_ISSUES[*]}" >&2
  exit 1
fi

exit 0
