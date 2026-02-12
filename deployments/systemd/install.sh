#!/usr/bin/env bash
set -euo pipefail

# Installs the systemd unit(s) from deployments/systemd/ into /etc/systemd/system.
#
# Default: installs ipfs-accelerate-mcp.service for the invoking user.
#
# Usage:
#   sudo deployments/systemd/install.sh
#   sudo deployments/systemd/install.sh --user barberb
#   sudo deployments/systemd/install.sh --unit ipfs-accelerate-mcp.service
#   sudo deployments/systemd/install.sh --unit ipfs-accelerate.service --no-start

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UNIT="ipfs-accelerate-mcp.service"
TARGET_DIR="/etc/systemd/system"
NO_START="0"

print_usage() {
  cat <<'EOF'
Usage: sudo deployments/systemd/install.sh [options]

Options:
  --unit NAME     Unit file to install (default: ipfs-accelerate-mcp.service)
  --user USER     Set User=/Group= in the unit (default: SUDO_USER, else current user)
  --no-start      Do not enable/restart the service
  -h, --help      Show help

Notes:
- Secrets should live in /etc/ipfs-accelerate/secrets.env (not in the unit file).
EOF
}

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: must be run as root (use sudo)" >&2
    exit 2
  fi
}

infer_user() {
  if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    echo "${SUDO_USER}"
    return 0
  fi
  # Fallback: if run as root without sudo context, use the first non-root login.
  echo "${USER:-root}"
}

infer_home() {
  local user="$1"
  local home=""

  home="$(getent passwd "${user}" 2>/dev/null | cut -d: -f6 || true)"
  if [[ -z "${home}" ]]; then
    home="/home/${user}"
  fi
  echo "${home}"
}

patch_unit_user_group() {
  local src="$1"
  local dst="$2"
  local user="$3"

  # Only patch when explicit user is provided.
  if [[ -z "${user}" ]]; then
    install -m 0644 "${src}" "${dst}"
    return 0
  fi

  local tmp
  tmp="$(mktemp)"
  trap 'rm -f "${tmp}"' RETURN

  local home
  home="$(infer_home "${user}")"

  sed \
    -e "s/^User=.*/User=${user}/" \
    -e "s/^Group=.*/Group=${user}/" \
    -e "s#%h#${home}#g" \
    "${src}" >"${tmp}"

  install -m 0644 "${tmp}" "${dst}"
}

ensure_secrets_env() {
  mkdir -p /etc/ipfs-accelerate
  chmod 0755 /etc/ipfs-accelerate

  if [[ ! -f /etc/ipfs-accelerate/secrets.env ]]; then
    cat > /etc/ipfs-accelerate/secrets.env <<'EOF'
# Optional secrets/env overrides.
# GH_TOKEN=...
# GITHUB_TOKEN=...
# IPFS_ACCELERATE_GITHUB_REPO=owner/repo
EOF
    chmod 0600 /etc/ipfs-accelerate/secrets.env
  fi
}

main() {
  require_root

  local user=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --unit)
        UNIT="$2"
        shift 2
        ;;
      --user)
        user="$2"
        shift 2
        ;;
      --no-start)
        NO_START="1"
        shift 1
        ;;
      -h|--help)
        print_usage
        exit 0
        ;;
      *)
        echo "ERROR: unknown arg: $1" >&2
        print_usage >&2
        exit 2
        ;;
    esac
  done

  if [[ -z "${user}" ]]; then
    user="$(infer_user)"
  fi

  local src_unit="${SCRIPT_DIR}/${UNIT}"
  if [[ ! -f "${src_unit}" ]]; then
    echo "ERROR: unit not found: ${src_unit}" >&2
    exit 2
  fi

  ensure_secrets_env

  mkdir -p "${TARGET_DIR}"
  patch_unit_user_group "${src_unit}" "${TARGET_DIR}/${UNIT}" "${user}"

  systemctl daemon-reload

  if [[ "${NO_START}" == "1" ]]; then
    echo "Installed ${UNIT} to ${TARGET_DIR}. Not enabling/starting (per --no-start)."
    exit 0
  fi

  systemctl enable "${UNIT}"
  systemctl restart "${UNIT}"

  echo "Installed + restarted ${UNIT}."
  echo "Follow logs: journalctl -u ${UNIT} -f"
}

main "$@"
