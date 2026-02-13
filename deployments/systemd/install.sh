#!/usr/bin/env bash
set -euo pipefail

# Installs the systemd unit(s) from deployments/systemd/ into /etc/systemd/system.
#
# Default: installs ipfs-accelerate.service for the invoking user.
#
# Usage:
#   sudo deployments/systemd/install.sh
#   sudo deployments/systemd/install.sh --user barberb
#   sudo deployments/systemd/install.sh --unit ipfs-accelerate-mcp.service
#   sudo deployments/systemd/install.sh --unit ipfs-accelerate.service --no-start

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Units explicitly requested via CLI flags. If none are provided, we default to
# installing the primary instance.
UNITS=()
TARGET_DIR="/etc/systemd/system"
NO_START="0"
INSTALL_BOTH="0"
PURGE_DROPINS="0"
SKIP_PY_DEPS="0"

print_usage() {
  cat <<'EOF'
Usage: sudo deployments/systemd/install.sh [options]

Options:
  --unit NAME     Unit file to install (repeatable; default: ipfs-accelerate.service)
  --both          Install both instances (ipfs-accelerate.service + ipfs-accelerate-mcp.service)
  --user USER     Set User=/Group= in the unit (default: SUDO_USER, else current user)
  --no-start      Do not enable/restart the service
  --purge-dropins Move aside existing /etc/systemd/system/<unit>.d drop-ins (backs up then removes)
  --skip-python-deps  Do not attempt to install Python deps into the target venv
  -h, --help      Show help

Notes:
- Secrets should live in /etc/ipfs-accelerate/secrets.env (not in the unit file).
- ipfs-accelerate.service runs the "standard" MCP implementation.
- ipfs-accelerate-mcp.service runs the MCP++ (Trio-oriented) implementation.
EOF
}

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: must be run as root (use sudo)" >&2
    exit 2
  fi
}

run_as_user() {
  local user="$1"; shift
  if command -v runuser >/dev/null 2>&1; then
    runuser -u "${user}" -- "$@"
    return $?
  fi
  su -s /bin/bash -c "$(printf '%q ' "$@")" "${user}"
}

ensure_python_deps() {
  local user="$1"
  local home="$2"
  local unit="$3"

  if [[ "${SKIP_PY_DEPS}" == "1" ]]; then
    return 0
  fi

  # MCP++ unit runs via Hypercorn w/ Trio workers.
  if [[ "${unit}" == "ipfs-accelerate-mcp.service" ]]; then
    local py="${PROJECT_DIR}/.venv/bin/python3"
    local pip="${PROJECT_DIR}/.venv/bin/pip"

    if [[ ! -x "${py}" || ! -x "${pip}" ]]; then
      echo "WARNING: venv not found at ${PROJECT_DIR}/.venv; cannot ensure hypercorn[trio]" >&2
      return 0
    fi

    if ! run_as_user "${user}" "${py}" -c 'import hypercorn' >/dev/null 2>&1; then
      echo "Installing hypercorn[trio] into ${PROJECT_DIR}/.venv (required for ipfs-accelerate-mcp.service)" >&2
      run_as_user "${user}" "${pip}" install -U 'hypercorn[trio]'
    fi
  fi
}

infer_user() {
  if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    echo "${SUDO_USER}"
    return 0
  fi

  # If run as root without sudo context, try to infer the active login user.
  # "logname" is usually the most reliable in interactive root shells.
  if command -v logname >/dev/null 2>&1; then
    local ln
    ln="$(logname 2>/dev/null || true)"
    if [[ -n "${ln}" && "${ln}" != "root" ]]; then
      echo "${ln}"
      return 0
    fi
  fi

  # If the installer lives under /home/<user>/..., infer <user>.
  if [[ "${SCRIPT_DIR}" =~ ^/home/([^/]+)/ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi

  # Last resort: USER (may be root).
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
  trap '[[ -n "${tmp:-}" ]] && rm -f "${tmp}"' RETURN

  local home
  home="$(infer_home "${user}")"

  sed \
    -e "s/^User=.*/User=${user}/" \
    -e "s/^Group=.*/Group=${user}/" \
    -e "s#%h/ipfs_accelerate_py#${PROJECT_DIR}#g" \
    -e "s#/root/ipfs_accelerate_py#${PROJECT_DIR}#g" \
    -e "s#/home/devel/ipfs_accelerate_py#${PROJECT_DIR}#g" \
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

# MCP++ HTTPS (Hypercorn)
# Default (auto-generated on first start):
# MCP_SSL_CERTFILE=%h/ipfs_accelerate_py/state/tls/mcpplusplus.crt
# MCP_SSL_KEYFILE=%h/ipfs_accelerate_py/state/tls/mcpplusplus.key
# Optional override for the SAN IP in the generated cert:
# MCP_LAN_IP_OVERRIDE=192.168.1.10
# If you later switch to a real domain and certbot:
# MCP_SSL_CERTFILE=/etc/letsencrypt/live/<your-domain>/fullchain.pem
# MCP_SSL_KEYFILE=/etc/letsencrypt/live/<your-domain>/privkey.pem
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
        if [[ "${INSTALL_BOTH}" == "1" ]]; then
          echo "ERROR: --unit cannot be used with --both" >&2
          exit 2
        fi
        UNITS+=("$2")
        shift 2
        ;;
      --both)
        if [[ "${#UNITS[@]}" -gt 0 ]]; then
          echo "ERROR: --both cannot be combined with --unit" >&2
          exit 2
        fi
        INSTALL_BOTH="1"
        UNITS=("ipfs-accelerate.service" "ipfs-accelerate-mcp.service")
        shift 1
        ;;
      --user)
        user="$2"
        shift 2
        ;;
      --no-start)
        NO_START="1"
        shift 1
        ;;
      --purge-dropins)
        PURGE_DROPINS="1"
        shift 1
        ;;
      --skip-python-deps)
        SKIP_PY_DEPS="1"
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

  if [[ "${#UNITS[@]}" -eq 0 ]]; then
    UNITS=("ipfs-accelerate.service")
  fi

  if [[ -z "${user}" ]]; then
    user="$(infer_user)"
  fi

  local home
  home="$(infer_home "${user}")"

  ensure_secrets_env

  mkdir -p "${TARGET_DIR}"
  for unit in "${UNITS[@]}"; do
    local src_unit="${SCRIPT_DIR}/${unit}"
    if [[ ! -f "${src_unit}" ]]; then
      echo "ERROR: unit not found: ${src_unit}" >&2
      exit 2
    fi

    if [[ "${PURGE_DROPINS}" == "1" ]]; then
      local dropin_dir="${TARGET_DIR}/${unit}.d"
      if [[ -d "${dropin_dir}" ]]; then
        local ts
        ts="$(date +%Y%m%d%H%M%S)"
        local backup_dir="${dropin_dir}.bak.${ts}"
        echo "Found drop-ins: ${dropin_dir}" >&2
        echo "Moving to backup: ${backup_dir}" >&2
        mv "${dropin_dir}" "${backup_dir}"
      fi
    fi

    patch_unit_user_group "${src_unit}" "${TARGET_DIR}/${unit}" "${user}"

    ensure_python_deps "${user}" "${home}" "${unit}"
  done

  systemctl daemon-reload

  if [[ "${NO_START}" == "1" ]]; then
    echo "Installed ${UNITS[*]} to ${TARGET_DIR}. Not enabling/starting (per --no-start)."
    exit 0
  fi

  for unit in "${UNITS[@]}"; do
    systemctl enable "${unit}"
    systemctl restart "${unit}"
    echo "Installed + restarted ${unit}."
    echo "Follow logs: journalctl -u ${unit} -f"
  done
}

main "$@"
