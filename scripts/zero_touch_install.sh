#!/usr/bin/env bash
# Zero-touch installer for ipfs_accelerate_py
# - Creates/updates .venv
# - Installs Python deps (no sudo)
# - Installs userland tools into ./bin when system install would require sudo
# - Ensures ./bin is on PATH (via shell rc modification)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$REPO_ROOT/bin"
VENV_DIR="$REPO_ROOT/.venv"

log() { echo "[install] $*"; }
warn() { echo "[install][WARN] $*" >&2; }

detect_shell_rc_files() {
  local files=()
  # Prefer more common interactive files; include only those that exist.
  [[ -f "$HOME/.bashrc" ]] && files+=("$HOME/.bashrc")
  [[ -f "$HOME/.zshrc" ]] && files+=("$HOME/.zshrc")
  [[ -f "$HOME/.profile" ]] && files+=("$HOME/.profile")
  # If none exist, create ~/.profile as a fallback.
  if [[ ${#files[@]} -eq 0 ]]; then
    touch "$HOME/.profile"
    files+=("$HOME/.profile")
  fi
  printf '%s\n' "${files[@]}"
}

ensure_repo_bin_on_path() {
  mkdir -p "$BIN_DIR"

  local marker_begin="# >>> ipfs_accelerate_py (repo-local bin) >>>"
  local marker_end="# <<< ipfs_accelerate_py (repo-local bin) <<<"
  local export_line="export PATH=\"$BIN_DIR:\${PATH}\""

  while IFS= read -r rc_file; do
    if grep -Fq "$marker_begin" "$rc_file"; then
      log "PATH hook already present in $rc_file"
      continue
    fi

    log "Adding PATH hook to $rc_file"
    {
      echo
      echo "$marker_begin"
      echo "$export_line"
      echo "$marker_end"
    } >> "$rc_file"
  done < <(detect_shell_rc_files)
}

create_or_update_venv() {
  mkdir -p "$BIN_DIR"

  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  else
    log "Using existing venv at $VENV_DIR"
  fi

  log "Upgrading pip/setuptools/wheel"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null
}

pip_install() {
  local requirements_file="$1"
  if [[ -f "$requirements_file" ]]; then
    log "Installing Python deps from $(realpath --relative-to="$REPO_ROOT" "$requirements_file" 2>/dev/null || echo "$requirements_file")"
    "$VENV_DIR/bin/python" -m pip install -r "$requirements_file"
  else
    warn "Requirements file missing: $requirements_file"
  fi
}

pip_install_editable_package() {
  # Avoid pulling in setup.py's very large dependency list.
  # The repo uses curated requirements files; install the package itself without deps.
  if [[ -f "$REPO_ROOT/setup.py" || -f "$REPO_ROOT/pyproject.toml" ]]; then
    log "Installing package in editable mode (no deps)"
    "$VENV_DIR/bin/python" -m pip install -e "$REPO_ROOT" --no-deps
  fi
}

install_gh_to_repo_bin_if_missing() {
  if command -v gh >/dev/null 2>&1; then
    log "gh already available on PATH ($(command -v gh))"
    return 0
  fi

  if [[ -x "$BIN_DIR/gh" ]]; then
    log "gh already installed in $BIN_DIR/gh"
    return 0
  fi

  local os arch asset_arch latest_version url tmp_dir
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"

  case "$os" in
    linux) ;;
    darwin) warn "Local gh install currently supports Linux only; skipping."; return 0 ;;
    *) warn "Unsupported OS for local gh install: $os; skipping."; return 0 ;;
  esac

  case "$arch" in
    x86_64|amd64) asset_arch="linux_amd64" ;;
    aarch64|arm64) asset_arch="linux_arm64" ;;
    *) warn "Unsupported architecture for local gh install: $arch; skipping."; return 0 ;;
  esac

  log "Installing gh CLI into $BIN_DIR (no sudo)"
  latest_version="$(curl -fsSL https://api.github.com/repos/cli/cli/releases/latest | grep '"tag_name"' | head -n1 | cut -d'"' -f4 | sed 's/^v//')"
  if [[ -z "$latest_version" ]]; then
    warn "Could not determine latest gh version; skipping."
    return 0
  fi

  url="https://github.com/cli/cli/releases/download/v${latest_version}/gh_${latest_version}_${asset_arch}.tar.gz"
  tmp_dir="$(mktemp -d)"

  ( 
    cd "$tmp_dir"
    curl -fL -o gh.tgz "$url"
    tar -xzf gh.tgz
    install -m 755 "gh_${latest_version}_${asset_arch}/bin/gh" "$BIN_DIR/gh"
  )

  rm -rf "$tmp_dir"
  log "Installed gh to $BIN_DIR/gh"
}

main() {
  log "Repo root: $REPO_ROOT"
  ensure_repo_bin_on_path
  create_or_update_venv

  # Install core deps first.
  pip_install "$REPO_ROOT/requirements.txt"

  # Install test/dev deps by default (can be disabled for minimal installs).
  if [[ "${INSTALL_TEST_DEPS:-1}" == "1" ]]; then
    pip_install "$REPO_ROOT/test/requirements.txt"
  else
    log "Skipping test dependencies (INSTALL_TEST_DEPS=0)"
  fi

  # Install curated base installer requirements if present.
  pip_install "$REPO_ROOT/install/requirements_base.txt"

  pip_install_editable_package

  # Install userland tools (no sudo) into ./bin when missing.
  install_gh_to_repo_bin_if_missing

  log "Done. Open a new shell (or 'source ~/.bashrc') to pick up $BIN_DIR on PATH."
  log "Python: $VENV_DIR/bin/python"
  log "Run tests: $VENV_DIR/bin/python -m pytest"
}

main "$@"
