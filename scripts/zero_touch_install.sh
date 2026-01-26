#!/usr/bin/env bash
# Zero-touch installer for ipfs_accelerate_py
# - Creates/updates .venv
# - Installs Python deps (no sudo)
# - Installs userland tools into ./bin when system install would require sudo
# - Ensures ./bin is on PATH (via shell rc modification)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${BIN_DIR:-$REPO_ROOT/bin}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
TEST_VENV_DIR="${TEST_VENV_DIR:-$REPO_ROOT/.venv-test}"

# Turnkey mode: a single knob that maps to sane defaults.
# Values: minimal | runtime | testing | full
# - minimal: runtime deps only, no tools, no test deps
# - runtime: runtime deps + curated base + local tools
# - testing: runtime + curated base + .[testing]
# - full: testing + heavy test venv + local tools
PROFILE="${PROFILE:-runtime}"

# Track whether the user explicitly set knobs (env or args) so profiles don't clobber them.
ENV_HAS_INSTALL_TEST_DEPS=0; [[ -n "${INSTALL_TEST_DEPS+x}" ]] && ENV_HAS_INSTALL_TEST_DEPS=1
ENV_HAS_INSTALL_HEAVY_TEST_DEPS=0; [[ -n "${INSTALL_HEAVY_TEST_DEPS+x}" ]] && ENV_HAS_INSTALL_HEAVY_TEST_DEPS=1
ENV_HAS_FORCE_LOCAL_TOOLS=0; [[ -n "${FORCE_LOCAL_TOOLS+x}" ]] && ENV_HAS_FORCE_LOCAL_TOOLS=1
ENV_HAS_OFFLINE=0; [[ -n "${OFFLINE+x}" ]] && ENV_HAS_OFFLINE=1
ENV_HAS_INSTALL_TOOL_GH=0; [[ -n "${INSTALL_TOOL_GH+x}" ]] && ENV_HAS_INSTALL_TOOL_GH=1
ENV_HAS_INSTALL_TOOL_IPFS=0; [[ -n "${INSTALL_TOOL_IPFS+x}" ]] && ENV_HAS_INSTALL_TOOL_IPFS=1
ENV_HAS_INSTALL_TOOL_JQ=0; [[ -n "${INSTALL_TOOL_JQ+x}" ]] && ENV_HAS_INSTALL_TOOL_JQ=1
ENV_HAS_INSTALL_TOOL_YQ=0; [[ -n "${INSTALL_TOOL_YQ+x}" ]] && ENV_HAS_INSTALL_TOOL_YQ=1
ENV_HAS_GH_VERSION=0; [[ -n "${GH_VERSION+x}" ]] && ENV_HAS_GH_VERSION=1
ENV_HAS_KUBO_VERSION=0; [[ -n "${KUBO_VERSION+x}" ]] && ENV_HAS_KUBO_VERSION=1
ENV_HAS_JQ_VERSION=0; [[ -n "${JQ_VERSION+x}" ]] && ENV_HAS_JQ_VERSION=1
ENV_HAS_YQ_VERSION=0; [[ -n "${YQ_VERSION+x}" ]] && ENV_HAS_YQ_VERSION=1

# Defaults: keep the main env consistent with runtime deps.
# Heavier / potentially conflicting stacks (selenium, big ML suites) are opt-in.
INSTALL_TEST_DEPS="${INSTALL_TEST_DEPS:-1}"                 # installs project extras: .[testing]
INSTALL_HEAVY_TEST_DEPS="${INSTALL_HEAVY_TEST_DEPS:-0}"     # installs test/requirements.txt into a separate venv

# Tool install behavior
FORCE_LOCAL_TOOLS="${FORCE_LOCAL_TOOLS:-0}"                 # install into ./bin even if tool exists on PATH
OFFLINE="${OFFLINE:-0}"                                     # skip network downloads

# pip behavior knobs (primarily for cross-platform install reliability)
PIP_PREFER_BINARY="${PIP_PREFER_BINARY:-1}"                 # 1 => pass --prefer-binary
PIP_RETRIES="${PIP_RETRIES:-5}"                             # pip network retries
PIP_TIMEOUT="${PIP_TIMEOUT:-60}"                            # pip network timeout (seconds)
PIP_ARGS="${PIP_ARGS:-}"                                    # additional args appended to pip install (e.g. '--extra-index-url ...')
PIP_NO_BUILD_ISOLATION="${PIP_NO_BUILD_ISOLATION:-0}"         # 1 => pass --no-build-isolation (useful for wheelhouse-only installs)

# wheel build / from-source flow
BUILD_WHEELS="${BUILD_WHEELS:-0}"                            # 1 => build wheels into wheelhouse before installing
USE_WHEELHOUSE="${USE_WHEELHOUSE:-0}"                        # 1 => install using --no-index --find-links wheelhouse
FROM_SOURCE="${FROM_SOURCE:-0}"                              # 1 => force sdist builds (pip --no-binary :all:) + build wheelhouse + use it
FROM_SOURCE_STRICT="${FROM_SOURCE_STRICT:-0}"                 # 1 => fail if any sdist build fails (no fallback to binary wheels)
PIP_NO_BINARY_ALL="${PIP_NO_BINARY_ALL:-0}"                   # 1 => add --no-binary :all: (forces source builds)
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-}"                          # if empty, derived from platform after detect_platform
REBUILD_WHEELS="${REBUILD_WHEELS:-0}"                          # 1 => always rebuild wheelhouse even if it already exists

# Tool versions ("latest" or explicit version like "0.39.0")
GH_VERSION="${GH_VERSION:-latest}"
KUBO_VERSION="${KUBO_VERSION:-latest}"
JQ_VERSION="${JQ_VERSION:-latest}"
YQ_VERSION="${YQ_VERSION:-latest}"
INSTALL_TOOL_GH="${INSTALL_TOOL_GH:-1}"
INSTALL_TOOL_IPFS="${INSTALL_TOOL_IPFS:-1}"
INSTALL_TOOL_JQ="${INSTALL_TOOL_JQ:-1}"
INSTALL_TOOL_YQ="${INSTALL_TOOL_YQ:-0}"

log() { echo "[install] $*"; }
warn() { echo "[install][WARN] $*" >&2; }

die() { echo "[install][ERROR] $*" >&2; exit 1; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

relpath_or_self() {
  local p="$1"
  if have_cmd realpath; then
    realpath --relative-to="$REPO_ROOT" "$p" 2>/dev/null || echo "$p"
  else
    echo "$p"
  fi
}

install_executable() {
  local src="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if have_cmd install; then
    install -m 755 "$src" "$dest"
  else
    cp "$src" "$dest"
    chmod 755 "$dest"
  fi
}

print_help() {
  cat <<EOF
Zero-touch installer for ipfs_accelerate_py

Usage:
  bash scripts/zero_touch_install.sh [options]

Options:
  --profile <minimal|runtime|testing|full>
  --offline                 (set OFFLINE=1)
  --force-local-tools        (set FORCE_LOCAL_TOOLS=1)
  --no-tools                 (disable gh/ipfs/jq/yq installs)
  --with-heavy-tests         (set INSTALL_HEAVY_TEST_DEPS=1)
  --without-tests            (set INSTALL_TEST_DEPS=0)
  --build-wheels             (set BUILD_WHEELS=1)
  --use-wheelhouse           (set USE_WHEELHOUSE=1; install only from wheelhouse)
  --from-source              (force source builds; implies --build-wheels + --use-wheelhouse)
  --from-source-strict        (like --from-source, but fail if any source build fails)
  --wheelhouse <dir>         (set WHEELHOUSE_DIR)
  --rebuild-wheels            (set REBUILD_WHEELS=1)
  -h, --help

Env overrides:
  BIN_DIR, VENV_DIR, TEST_VENV_DIR
  PROFILE
  OFFLINE, FORCE_LOCAL_TOOLS
  PIP_PREFER_BINARY, PIP_RETRIES, PIP_TIMEOUT, PIP_ARGS
  BUILD_WHEELS, USE_WHEELHOUSE, FROM_SOURCE, FROM_SOURCE_STRICT, PIP_NO_BINARY_ALL, WHEELHOUSE_DIR, REBUILD_WHEELS
  PIP_NO_BUILD_ISOLATION
  INSTALL_TEST_DEPS, INSTALL_HEAVY_TEST_DEPS
  INSTALL_TOOL_GH, INSTALL_TOOL_IPFS, INSTALL_TOOL_JQ, INSTALL_TOOL_YQ
  GH_VERSION, KUBO_VERSION, JQ_VERSION, YQ_VERSION ("latest" or explicit version)
EOF
}

parse_args() {
  local arg
  while [[ $# -gt 0 ]]; do
    arg="$1"
    case "$arg" in
      --profile)
        shift
        [[ $# -gt 0 ]] || die "--profile requires a value"
        PROFILE="$1"
        ;;
      --offline)
        OFFLINE=1
        ENV_HAS_OFFLINE=1
        ;;
      --force-local-tools)
        FORCE_LOCAL_TOOLS=1
        ENV_HAS_FORCE_LOCAL_TOOLS=1
        ;;
      --no-tools)
        INSTALL_TOOL_GH=0; ENV_HAS_INSTALL_TOOL_GH=1
        INSTALL_TOOL_IPFS=0; ENV_HAS_INSTALL_TOOL_IPFS=1
        INSTALL_TOOL_JQ=0; ENV_HAS_INSTALL_TOOL_JQ=1
        INSTALL_TOOL_YQ=0; ENV_HAS_INSTALL_TOOL_YQ=1
        ;;
      --with-heavy-tests)
        INSTALL_HEAVY_TEST_DEPS=1
        ENV_HAS_INSTALL_HEAVY_TEST_DEPS=1
        ;;
      --without-tests)
        INSTALL_TEST_DEPS=0
        ENV_HAS_INSTALL_TEST_DEPS=1
        ;;
      --build-wheels)
        BUILD_WHEELS=1
        ;;
      --use-wheelhouse)
        USE_WHEELHOUSE=1
        ;;
      --from-source)
        FROM_SOURCE=1
        ;;
      --from-source-strict)
        FROM_SOURCE=1
        FROM_SOURCE_STRICT=1
        ;;
      --wheelhouse)
        shift
        [[ $# -gt 0 ]] || die "--wheelhouse requires a value"
        WHEELHOUSE_DIR="$1"
        ;;
      --rebuild-wheels)
        REBUILD_WHEELS=1
        ;;
      -h|--help)
        print_help
        exit 0
        ;;
      *)
        die "Unknown option: $arg (use --help)"
        ;;
    esac
    shift
  done
}

apply_profile_defaults() {
  case "$PROFILE" in
    minimal)
      [[ $ENV_HAS_INSTALL_TEST_DEPS -eq 1 ]] || INSTALL_TEST_DEPS=0
      [[ $ENV_HAS_INSTALL_HEAVY_TEST_DEPS -eq 1 ]] || INSTALL_HEAVY_TEST_DEPS=0
      [[ $ENV_HAS_INSTALL_TOOL_GH -eq 1 ]] || INSTALL_TOOL_GH=0
      [[ $ENV_HAS_INSTALL_TOOL_IPFS -eq 1 ]] || INSTALL_TOOL_IPFS=0
      [[ $ENV_HAS_INSTALL_TOOL_JQ -eq 1 ]] || INSTALL_TOOL_JQ=0
      [[ $ENV_HAS_INSTALL_TOOL_YQ -eq 1 ]] || INSTALL_TOOL_YQ=0
      ;;
    runtime)
      [[ $ENV_HAS_INSTALL_TEST_DEPS -eq 1 ]] || INSTALL_TEST_DEPS=0
      [[ $ENV_HAS_INSTALL_HEAVY_TEST_DEPS -eq 1 ]] || INSTALL_HEAVY_TEST_DEPS=0
      [[ $ENV_HAS_INSTALL_TOOL_GH -eq 1 ]] || INSTALL_TOOL_GH=1
      [[ $ENV_HAS_INSTALL_TOOL_IPFS -eq 1 ]] || INSTALL_TOOL_IPFS=1
      [[ $ENV_HAS_INSTALL_TOOL_JQ -eq 1 ]] || INSTALL_TOOL_JQ=1
      [[ $ENV_HAS_INSTALL_TOOL_YQ -eq 1 ]] || INSTALL_TOOL_YQ=0
      ;;
    testing)
      [[ $ENV_HAS_INSTALL_TEST_DEPS -eq 1 ]] || INSTALL_TEST_DEPS=1
      [[ $ENV_HAS_INSTALL_HEAVY_TEST_DEPS -eq 1 ]] || INSTALL_HEAVY_TEST_DEPS=0
      [[ $ENV_HAS_INSTALL_TOOL_GH -eq 1 ]] || INSTALL_TOOL_GH=1
      [[ $ENV_HAS_INSTALL_TOOL_IPFS -eq 1 ]] || INSTALL_TOOL_IPFS=1
      [[ $ENV_HAS_INSTALL_TOOL_JQ -eq 1 ]] || INSTALL_TOOL_JQ=1
      [[ $ENV_HAS_INSTALL_TOOL_YQ -eq 1 ]] || INSTALL_TOOL_YQ=1
      ;;
    full)
      [[ $ENV_HAS_INSTALL_TEST_DEPS -eq 1 ]] || INSTALL_TEST_DEPS=1
      [[ $ENV_HAS_INSTALL_HEAVY_TEST_DEPS -eq 1 ]] || INSTALL_HEAVY_TEST_DEPS=1
      [[ $ENV_HAS_INSTALL_TOOL_GH -eq 1 ]] || INSTALL_TOOL_GH=1
      [[ $ENV_HAS_INSTALL_TOOL_IPFS -eq 1 ]] || INSTALL_TOOL_IPFS=1
      [[ $ENV_HAS_INSTALL_TOOL_JQ -eq 1 ]] || INSTALL_TOOL_JQ=1
      [[ $ENV_HAS_INSTALL_TOOL_YQ -eq 1 ]] || INSTALL_TOOL_YQ=1
      ;;
    *)
      die "Unknown PROFILE='$PROFILE' (use minimal|runtime|testing|full)"
      ;;
  esac

  # FROM_SOURCE is an explicit request to build everything from sdists.
  # It implies wheelhouse build + wheelhouse-only installs and disables prefer-binary.
  if [[ "$FROM_SOURCE" == "1" ]]; then
    PIP_NO_BINARY_ALL=1
    PIP_PREFER_BINARY=0
    BUILD_WHEELS=1
    USE_WHEELHOUSE=1
  fi

  # Wheelhouse-only installs should avoid build isolation, since it would try to
  # download build requirements (setuptools/wheel/etc) from the index.
  if [[ "$USE_WHEELHOUSE" == "1" ]]; then
    PIP_NO_BUILD_ISOLATION=1
  fi
}

print_config() {
  log "Config: PROFILE=$PROFILE BIN_DIR=$BIN_DIR VENV_DIR=$VENV_DIR TEST_VENV_DIR=$TEST_VENV_DIR"
  log "Config: OFFLINE=$OFFLINE FORCE_LOCAL_TOOLS=$FORCE_LOCAL_TOOLS"
  log "Config: pip prefer-binary=$PIP_PREFER_BINARY retries=$PIP_RETRIES timeout=$PIP_TIMEOUT no-build-isolation=$PIP_NO_BUILD_ISOLATION${PIP_ARGS:+ args='$PIP_ARGS'}"
  log "Config: wheels build=$BUILD_WHEELS use-wheelhouse=$USE_WHEELHOUSE from-source=$FROM_SOURCE strict=$FROM_SOURCE_STRICT no-binary-all=$PIP_NO_BINARY_ALL rebuild=$REBUILD_WHEELS${WHEELHOUSE_DIR:+ wheelhouse='$WHEELHOUSE_DIR'}"
  log "Config: INSTALL_TEST_DEPS=$INSTALL_TEST_DEPS INSTALL_HEAVY_TEST_DEPS=$INSTALL_HEAVY_TEST_DEPS"
  log "Config: tools gh=$INSTALL_TOOL_GH ipfs=$INSTALL_TOOL_IPFS jq=$INSTALL_TOOL_JQ yq=$INSTALL_TOOL_YQ"
}

PLATFORM_OS=""
PLATFORM_ARCH=""
PLATFORM_LIBC=""
PLATFORM_ENV=""

detect_platform() {
  local os arch
  os="$(uname -s 2>/dev/null | tr '[:upper:]' '[:lower:]' || echo unknown)"
  arch="$(uname -m 2>/dev/null || echo unknown)"

  PLATFORM_OS="$os"
  case "$arch" in
    x86_64|amd64) PLATFORM_ARCH="amd64" ;;
    aarch64|arm64) PLATFORM_ARCH="arm64" ;;
    armv7l|armv7) PLATFORM_ARCH="armv7" ;;
    i386|i686) PLATFORM_ARCH="386" ;;
    *) PLATFORM_ARCH="$arch" ;;
  esac

  PLATFORM_ENV=""
  if [[ "$PLATFORM_OS" == "linux" ]] && [[ -r /proc/version ]] && grep -qi microsoft /proc/version; then
    PLATFORM_ENV="wsl"
  fi

  PLATFORM_LIBC="unknown"
  if [[ "$PLATFORM_OS" == "linux" ]]; then
    if have_cmd ldd && ldd --version 2>&1 | grep -qi musl; then
      PLATFORM_LIBC="musl"
    elif have_cmd ldd && ldd --version 2>&1 | grep -qiE 'glibc|gnu libc'; then
      PLATFORM_LIBC="glibc"
    elif [[ -e /lib/ld-musl-aarch64.so.1 || -e /lib/ld-musl-x86_64.so.1 ]]; then
      PLATFORM_LIBC="musl"
    else
      PLATFORM_LIBC="glibc"
    fi
  fi

  log "Platform: os=$PLATFORM_OS arch=$PLATFORM_ARCH libc=$PLATFORM_LIBC${PLATFORM_ENV:+ env=$PLATFORM_ENV}"
}

download_to() {
  local url="$1" dest="$2"
  if [[ "$OFFLINE" == "1" ]]; then
    warn "OFFLINE=1; skipping download: $url"
    return 1
  fi

  if have_cmd curl; then
    curl -fL --retry 3 --retry-delay 1 -o "$dest" "$url"
    return 0
  fi
  if have_cmd wget; then
    wget -O "$dest" "$url"
    return 0
  fi

  if have_cmd python3; then
    python3 - "$url" "$dest" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
dest = sys.argv[2]
with urllib.request.urlopen(url) as r, open(dest, 'wb') as f:
    f.write(r.read())
PY
    return 0
  fi

  warn "No downloader available (need curl/wget/python3)."
  return 1
}

extract_tgz() {
  local archive="$1" dest_dir="$2"
  mkdir -p "$dest_dir"
  if have_cmd tar; then
    tar -xzf "$archive" -C "$dest_dir"
    return 0
  fi
  if have_cmd python3; then
    python3 - "$archive" "$dest_dir" <<'PY'
import sys
import tarfile

archive = sys.argv[1]
dest = sys.argv[2]
with tarfile.open(archive, 'r:gz') as tf:
    tf.extractall(dest)
PY
    return 0
  fi
  warn "Cannot extract $archive (need tar or python3)."
  return 1
}

github_latest_version() {
  local owner="$1" repo="$2"
  if have_cmd python3; then
    python3 - "$owner" "$repo" <<'PY'
import json
import sys
import urllib.request

owner, repo = sys.argv[1], sys.argv[2]
url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
req = urllib.request.Request(url, headers={"User-Agent": "ipfs-accelerate-zero-touch"})
with urllib.request.urlopen(req) as r:
    data = json.load(r)
tag = data.get("tag_name", "")
print(tag[1:] if tag.startswith('v') else tag)
PY
    return 0
  fi

  if have_cmd curl; then
    curl -fsSL "https://api.github.com/repos/${owner}/${repo}/releases/latest" | grep '"tag_name"' | head -n1 | cut -d'"' -f4 | sed 's/^v//'
    return 0
  fi

  warn "Cannot query GitHub latest version (need python3 or curl)."
  return 1
}

should_use_system_tool() {
  local tool="$1"
  if [[ "$FORCE_LOCAL_TOOLS" == "1" ]]; then
    return 1
  fi
  command -v "$tool" >/dev/null 2>&1
}

ensure_prereqs() {
  if ! have_cmd python3; then
    warn "python3 is required but not found on PATH."
    exit 1
  fi
  if ! have_cmd git; then
    warn "git is required (requirements include git URLs), but not found on PATH."
    exit 1
  fi
}

warn_if_build_tools_missing() {
  # We can't install system toolchains without sudo, but we can give actionable hints.
  case "$PLATFORM_OS" in
    linux)
      if ! have_cmd gcc && ! have_cmd clang; then
        warn "No C compiler found (gcc/clang). Some Python packages may require a compiler to build wheels."
      fi
      if ! have_cmd make; then
        warn "'make' not found. Some Python packages may require build tools."
      fi
      if ! have_cmd pkg-config; then
        warn "'pkg-config' not found. Some optional dependencies may fail to build from source."
      fi
      ;;
    darwin)
      if have_cmd xcode-select; then
        if ! xcode-select -p >/dev/null 2>&1; then
          warn "Xcode Command Line Tools not configured. Run: xcode-select --install"
        fi
      else
        warn "xcode-select not found; ensure Xcode Command Line Tools are installed."
      fi
      ;;
    msys*|mingw*|cygwin*)
      warn "Detected $PLATFORM_OS environment. This bash installer is primarily for Linux/macOS; consider using WSL on Windows for best results."
      ;;
  esac

  if ! have_cmd rustc || ! have_cmd cargo; then
    warn "Rust toolchain not found (rustc/cargo). Some dependencies may require it to build from source on certain platforms."
  fi
}

pip_install_with_common_args() {
  local venv_dir="$1"; shift
  local -a cmd=("$venv_dir/bin/python" -m pip install)

  if [[ "$PIP_PREFER_BINARY" == "1" ]]; then
    cmd+=(--prefer-binary)
  fi
  if [[ -n "$PIP_TIMEOUT" ]]; then
    cmd+=(--timeout "$PIP_TIMEOUT")
  fi
  if [[ -n "$PIP_RETRIES" ]]; then
    cmd+=(--retries "$PIP_RETRIES")
  fi
  if [[ -n "${PIP_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local -a extra=( $PIP_ARGS )
    cmd+=("${extra[@]}")
  fi

  if [[ "$PIP_NO_BUILD_ISOLATION" == "1" ]]; then
    cmd+=(--no-build-isolation)
  fi

  if [[ "$PIP_NO_BINARY_ALL" == "1" ]]; then
    cmd+=(--no-binary :all:)
  fi

  if [[ "$USE_WHEELHOUSE" == "1" ]]; then
    [[ -n "${WHEELHOUSE_DIR:-}" ]] || die "USE_WHEELHOUSE=1 but WHEELHOUSE_DIR is empty"
    cmd+=(--no-index --find-links "$WHEELHOUSE_DIR")
  fi

  cmd+=("$@")
  "${cmd[@]}"
}

pip_install_requirements_wheelhouse_aware() {
  local venv_dir="$1" requirements_file="$2"
  local constraints_file=""

  # Prevent the curated base set from upgrading runtime-critical pins.
  # In particular: ipfshttpclient requires urllib3<2.
  if [[ "$venv_dir" == "$VENV_DIR" && "$requirements_file" == "$REPO_ROOT/install/requirements_base.txt" ]]; then
    constraints_file="$(mktemp)"
    echo "urllib3<2" >"$constraints_file"
  fi

  # Normal path: just install the requirements file.
  if [[ "$USE_WHEELHOUSE" != "1" ]]; then
    if [[ -n "$constraints_file" ]]; then
      pip_install_with_common_args "$venv_dir" -c "$constraints_file" -r "$requirements_file"
      rm -f "$constraints_file"
    else
      pip_install_with_common_args "$venv_dir" -r "$requirements_file"
    fi
    return 0
  fi

  # Wheelhouse-only mode cannot satisfy VCS requirements (git+) without hitting
  # the network and/or build isolation. We build those wheels ahead of time into
  # the wheelhouse, then install by distribution name from wheelhouse.
  local tmp_req vcs_name
  tmp_req="$(mktemp)"
  : >"$tmp_req"
  local -a vcs_pkgs=()

  while IFS= read -r line || [[ -n "$line" ]]; do
    # strip comments and whitespace
    line="${line%%#*}"
    line="${line//$'\r'/}"
    line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$line" ]] && continue

    vcs_name=""
    if [[ "$line" =~ ^([A-Za-z0-9_.-]+)[[:space:]]*@[[:space:]]*git\+ ]]; then
      vcs_name="${BASH_REMATCH[1]}"
    elif [[ "$line" == git+* && "$line" == *"#egg="* ]]; then
      vcs_name="${line##*#egg=}"
      vcs_name="${vcs_name%%&*}"
    fi

    if [[ -n "$vcs_name" ]]; then
      vcs_pkgs+=("$vcs_name")
    else
      echo "$line" >>"$tmp_req"
    fi
  done <"$requirements_file"

  if [[ -s "$tmp_req" ]]; then
    if [[ -n "$constraints_file" ]]; then
      pip_install_with_common_args "$venv_dir" -c "$constraints_file" -r "$tmp_req"
    else
      pip_install_with_common_args "$venv_dir" -r "$tmp_req"
    fi
  fi

  rm -f "$tmp_req"

  if [[ ${#vcs_pkgs[@]} -gt 0 ]]; then
    log "Wheelhouse mode: installing VCS deps from wheelhouse: ${vcs_pkgs[*]}"
    if [[ -n "$constraints_file" ]]; then
      pip_install_with_common_args "$venv_dir" -c "$constraints_file" "${vcs_pkgs[@]}"
    else
      pip_install_with_common_args "$venv_dir" "${vcs_pkgs[@]}"
    fi
  fi

  if [[ -n "$constraints_file" ]]; then
    rm -f "$constraints_file"
  fi

  return 0
}

pip_wheel_with_common_args() {
  local venv_dir="$1"; shift
  local -a cmd=("$venv_dir/bin/python" -m pip wheel)

  if [[ -n "$PIP_TIMEOUT" ]]; then
    cmd+=(--timeout "$PIP_TIMEOUT")
  fi
  if [[ -n "$PIP_RETRIES" ]]; then
    cmd+=(--retries "$PIP_RETRIES")
  fi
  if [[ -n "${PIP_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local -a extra=( $PIP_ARGS )
    cmd+=("${extra[@]}")
  fi

  if [[ "$PIP_NO_BINARY_ALL" == "1" ]]; then
    cmd+=(--no-binary :all:)
  fi

  cmd+=("$@")
  "${cmd[@]}"
}

derive_wheelhouse_dir_if_needed() {
  if [[ -z "${WHEELHOUSE_DIR:-}" ]]; then
    WHEELHOUSE_DIR="$REPO_ROOT/wheelhouse/${PLATFORM_OS}-${PLATFORM_ARCH}-${PLATFORM_LIBC}"
  fi
}

build_wheelhouse() {
  local venv_dir="$1"
  derive_wheelhouse_dir_if_needed
  mkdir -p "$WHEELHOUSE_DIR"

  if [[ "$REBUILD_WHEELS" != "1" ]]; then
    if ls "$WHEELHOUSE_DIR"/*.whl >/dev/null 2>&1; then
      log "Wheelhouse already has wheels; skipping rebuild (set REBUILD_WHEELS=1 or use --rebuild-wheels to force)"
      return 0
    fi
  fi

  log "Building wheelhouse at $(relpath_or_self "$WHEELHOUSE_DIR")"

  if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
    log "Wheel: requirements.txt"
    if ! pip_wheel_with_common_args "$venv_dir" -r "$REPO_ROOT/requirements.txt" -w "$WHEELHOUSE_DIR"; then
      if [[ "$FROM_SOURCE_STRICT" == "1" ]]; then
        die "From-source strict mode: failed building wheels for requirements.txt"
      fi
      warn "From-source build failed for requirements.txt; falling back to binary wheels to keep install working."
      local saved_no_binary="$PIP_NO_BINARY_ALL"
      PIP_NO_BINARY_ALL=0
      pip_wheel_with_common_args "$venv_dir" -r "$REPO_ROOT/requirements.txt" -w "$WHEELHOUSE_DIR"
      PIP_NO_BINARY_ALL="$saved_no_binary"
    fi
  fi

  if [[ -f "$REPO_ROOT/install/requirements_base.txt" ]]; then
    log "Wheel: install/requirements_base.txt"
    # Keep urllib3 pinned <2 for the runtime wheelhouse.
    local base_constraints
    base_constraints="$(mktemp)"
    echo "urllib3<2" >"$base_constraints"
    if ! pip_wheel_with_common_args "$venv_dir" -c "$base_constraints" -r "$REPO_ROOT/install/requirements_base.txt" -w "$WHEELHOUSE_DIR"; then
      if [[ "$FROM_SOURCE_STRICT" == "1" ]]; then
        rm -f "$base_constraints"
        die "From-source strict mode: failed building wheels for install/requirements_base.txt"
      fi
      warn "From-source build failed for install/requirements_base.txt; falling back to binary wheels to keep install working."
      local saved_no_binary="$PIP_NO_BINARY_ALL"
      PIP_NO_BINARY_ALL=0
      pip_wheel_with_common_args "$venv_dir" -c "$base_constraints" -r "$REPO_ROOT/install/requirements_base.txt" -w "$WHEELHOUSE_DIR"
      PIP_NO_BINARY_ALL="$saved_no_binary"
    fi
    rm -f "$base_constraints"
  fi

  if [[ "$INSTALL_TEST_DEPS" == "1" ]]; then
    log "Wheel: project[testing] extras"
    if ! pip_wheel_with_common_args "$venv_dir" "$REPO_ROOT[testing]" -w "$WHEELHOUSE_DIR"; then
      if [[ "$FROM_SOURCE_STRICT" == "1" ]]; then
        die "From-source strict mode: failed building wheels for project[testing] extras"
      fi
      warn "From-source build failed for project[testing] extras; falling back to binary wheels to keep install working."
      local saved_no_binary="$PIP_NO_BINARY_ALL"
      PIP_NO_BINARY_ALL=0
      if ! pip_wheel_with_common_args "$venv_dir" "$REPO_ROOT[testing]" -w "$WHEELHOUSE_DIR"; then
        warn "Failed to build wheels for project[testing] extras even with binary fallback; continuing."
      fi
      PIP_NO_BINARY_ALL="$saved_no_binary"
    fi
  fi

  if [[ "$INSTALL_HEAVY_TEST_DEPS" == "1" && -f "$REPO_ROOT/test/requirements.txt" ]]; then
    log "Wheel: test/requirements.txt"
    if ! pip_wheel_with_common_args "$venv_dir" -r "$REPO_ROOT/test/requirements.txt" -w "$WHEELHOUSE_DIR"; then
      if [[ "$FROM_SOURCE_STRICT" == "1" ]]; then
        die "From-source strict mode: failed building wheels for test/requirements.txt"
      fi
      warn "From-source build failed for test/requirements.txt; falling back to binary wheels to keep install working."
      local saved_no_binary="$PIP_NO_BINARY_ALL"
      PIP_NO_BINARY_ALL=0
      pip_wheel_with_common_args "$venv_dir" -r "$REPO_ROOT/test/requirements.txt" -w "$WHEELHOUSE_DIR"
      PIP_NO_BINARY_ALL="$saved_no_binary"
    fi
  fi
}

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
  local venv_dir="$1"
  mkdir -p "$BIN_DIR"

  if [[ ! -d "$venv_dir" ]]; then
    log "Creating venv at $venv_dir"
    python3 -m venv "$venv_dir"
  else
    log "Using existing venv at $venv_dir"
  fi

  log "Upgrading pip/setuptools/wheel ($venv_dir)"
  # In wheelhouse-only or offline contexts, upgrading may be impossible; don't
  # fail the whole install over it.
  if [[ "$OFFLINE" == "1" || "$USE_WHEELHOUSE" == "1" ]]; then
    if ! pip_install_with_common_args "$venv_dir" --upgrade pip setuptools wheel >/dev/null 2>&1; then
      warn "Could not upgrade pip/setuptools/wheel (offline/wheelhouse mode); continuing with bundled versions."
    fi
  else
    "$venv_dir/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null
  fi
}

pip_install() {
  local venv_dir="$1"
  local requirements_file="$2"
  if [[ -f "$requirements_file" ]]; then
    log "Installing Python deps from $(relpath_or_self "$requirements_file")"
    pip_install_requirements_wheelhouse_aware "$venv_dir" "$requirements_file"
  else
    warn "Requirements file missing: $requirements_file"
  fi
}

pip_install_editable_package() {
  # Avoid pulling in setup.py's very large dependency list.
  # The repo uses curated requirements files; install the package itself without deps.
  if [[ -f "$REPO_ROOT/setup.py" || -f "$REPO_ROOT/pyproject.toml" ]]; then
    log "Installing package in editable mode (no deps)"
    pip_install_with_common_args "$1" -e "$REPO_ROOT" --no-deps
  fi
}

pip_install_extras() {
  local venv_dir="$1"
  local extras="$2"
  log "Installing package extras: $extras"
  pip_install_with_common_args "$venv_dir" -e "$REPO_ROOT[$extras]"
}

pip_check() {
  local venv_dir="$1"
  log "Checking dependency health ($venv_dir)"
  # pip check returns non-zero for conflicts; we surface as WARN to keep installer usable.
  if ! "$venv_dir/bin/python" -m pip check; then
    warn "Dependency conflicts detected in $venv_dir (see output above)"
  fi
}

install_gh_to_repo_bin_if_missing() {
  if [[ "$INSTALL_TOOL_GH" != "1" ]]; then
    log "Skipping gh install (INSTALL_TOOL_GH=$INSTALL_TOOL_GH)"
    return 0
  fi

  if should_use_system_tool gh; then
    log "gh already available on PATH ($(command -v gh))"
    return 0
  fi

  if [[ -x "$BIN_DIR/gh" ]]; then
    log "gh already installed in $BIN_DIR/gh"
    return 0
  fi

  local asset_arch version url tmp_dir unpack_dir
  case "$PLATFORM_OS" in
    linux) asset_arch="linux_${PLATFORM_ARCH}" ;;
    darwin) asset_arch="macOS_${PLATFORM_ARCH}" ;;
    *) warn "Unsupported OS for local gh install: $PLATFORM_OS; skipping."; return 0 ;;
  esac

  case "$PLATFORM_ARCH" in
    amd64|arm64) ;;
    *) warn "Unsupported architecture for local gh install: $PLATFORM_ARCH; skipping."; return 0 ;;
  esac

  log "Installing gh CLI into $BIN_DIR (no sudo)"

  if [[ "$GH_VERSION" == "latest" ]]; then
    version="$(github_latest_version cli cli || true)"
  else
    version="$GH_VERSION"
  fi
  if [[ -z "$version" ]]; then
    warn "Could not determine gh version; skipping."
    return 0
  fi

  url="https://github.com/cli/cli/releases/download/v${version}/gh_${version}_${asset_arch}.tar.gz"
  tmp_dir="$(mktemp -d)"

  (
    cd "$tmp_dir"
    download_to "$url" gh.tgz
    extract_tgz gh.tgz "$tmp_dir"
    unpack_dir="gh_${version}_${asset_arch}"
    install_executable "$unpack_dir/bin/gh" "$BIN_DIR/gh"
  )

  rm -rf "$tmp_dir"
  log "Installed gh to $BIN_DIR/gh"
}

install_ipfs_kubo_to_repo_bin_if_missing() {
  if [[ "$INSTALL_TOOL_IPFS" != "1" ]]; then
    log "Skipping ipfs install (INSTALL_TOOL_IPFS=$INSTALL_TOOL_IPFS)"
    return 0
  fi

  if should_use_system_tool ipfs; then
    log "ipfs already available on PATH ($(command -v ipfs))"
    return 0
  fi
  if [[ -x "$BIN_DIR/ipfs" ]]; then
    log "ipfs already installed in $BIN_DIR/ipfs"
    return 0
  fi

  local asset_arch version url tmp_dir os_tag
  case "$PLATFORM_OS" in
    linux) os_tag="linux" ;;
    darwin) os_tag="darwin" ;;
    *) warn "Unsupported OS for local ipfs install: $PLATFORM_OS; skipping."; return 0 ;;
  esac

  case "$PLATFORM_ARCH" in
    amd64) asset_arch="amd64" ;;
    arm64) asset_arch="arm64" ;;
    *) warn "Unsupported architecture for local ipfs install: $PLATFORM_ARCH; skipping."; return 0 ;;
  esac

  if [[ "$PLATFORM_OS" == "linux" && "$PLATFORM_LIBC" == "musl" ]]; then
    warn "Detected musl libc; upstream kubo binaries may be incompatible."
  fi

  log "Installing ipfs (kubo) into $BIN_DIR (no sudo)"

  if [[ "$KUBO_VERSION" == "latest" ]]; then
    version="$(github_latest_version ipfs kubo || true)"
  else
    version="$KUBO_VERSION"
  fi
  if [[ -z "$version" ]]; then
    warn "Could not determine kubo version; skipping."
    return 0
  fi

  url="https://github.com/ipfs/kubo/releases/download/v${version}/kubo_v${version}_${os_tag}-${asset_arch}.tar.gz"
  tmp_dir="$(mktemp -d)"
  (
    cd "$tmp_dir"
    download_to "$url" kubo.tgz
    extract_tgz kubo.tgz "$tmp_dir"
    install_executable "kubo/ipfs" "$BIN_DIR/ipfs"
  )
  rm -rf "$tmp_dir"
  log "Installed ipfs to $BIN_DIR/ipfs"
}

install_jq_to_repo_bin_if_missing() {
  if [[ "$INSTALL_TOOL_JQ" != "1" ]]; then
    log "Skipping jq install (INSTALL_TOOL_JQ=$INSTALL_TOOL_JQ)"
    return 0
  fi

  if should_use_system_tool jq; then
    log "jq already available on PATH ($(command -v jq))"
    return 0
  fi
  if [[ -x "$BIN_DIR/jq" ]]; then
    log "jq already installed in $BIN_DIR/jq"
    return 0
  fi

  local os_tag arch_name url tmp_file

  case "$PLATFORM_OS" in
    linux) os_tag="linux" ;;
    darwin) os_tag="macos" ;;
    *) warn "Unsupported OS for local jq install: $PLATFORM_OS; skipping."; return 0 ;;
  esac

  case "$PLATFORM_ARCH" in
    amd64) arch_name="amd64" ;;
    arm64) arch_name="arm64" ;;
    *) warn "Unsupported architecture for local jq install: $PLATFORM_ARCH; skipping."; return 0 ;;
  esac

  # Prefer the new jqlang/jq release artifacts.
  url="https://github.com/jqlang/jq/releases/latest/download/jq-${os_tag}-${arch_name}"
  tmp_file="$(mktemp)"
  log "Installing jq into $BIN_DIR (no sudo)"
  download_to "$url" "$tmp_file"
  install_executable "$tmp_file" "$BIN_DIR/jq"
  rm -f "$tmp_file"
  log "Installed jq to $BIN_DIR/jq"
}

install_yq_to_repo_bin_if_missing() {
  if [[ "$INSTALL_TOOL_YQ" != "1" ]]; then
    log "Skipping yq install (INSTALL_TOOL_YQ=$INSTALL_TOOL_YQ)"
    return 0
  fi

  if should_use_system_tool yq; then
    log "yq already available on PATH ($(command -v yq))"
    return 0
  fi
  if [[ -x "$BIN_DIR/yq" ]]; then
    log "yq already installed in $BIN_DIR/yq"
    return 0
  fi

  local os_tag arch_name version url tmp_file

  case "$PLATFORM_OS" in
    linux) os_tag="linux" ;;
    darwin) os_tag="darwin" ;;
    *) warn "Unsupported OS for local yq install: $PLATFORM_OS; skipping."; return 0 ;;
  esac

  case "$PLATFORM_ARCH" in
    amd64) arch_name="amd64" ;;
    arm64) arch_name="arm64" ;;
    *) warn "Unsupported architecture for local yq install: $PLATFORM_ARCH; skipping."; return 0 ;;
  esac

  if [[ "$YQ_VERSION" == "latest" ]]; then
    version="$(github_latest_version mikefarah yq || true)"
  else
    version="$YQ_VERSION"
  fi
  if [[ -z "$version" ]]; then
    warn "Could not determine yq version; skipping."
    return 0
  fi

  url="https://github.com/mikefarah/yq/releases/download/v${version}/yq_${os_tag}_${arch_name}"
  tmp_file="$(mktemp)"
  log "Installing yq into $BIN_DIR (no sudo)"
  download_to "$url" "$tmp_file"
  install_executable "$tmp_file" "$BIN_DIR/yq"
  rm -f "$tmp_file"
  log "Installed yq to $BIN_DIR/yq"
}

main() {
  log "Repo root: $REPO_ROOT"
  parse_args "$@"
  apply_profile_defaults
  print_config
  detect_platform
  derive_wheelhouse_dir_if_needed
  ensure_prereqs
  warn_if_build_tools_missing
  ensure_repo_bin_on_path
  create_or_update_venv "$VENV_DIR"

  if [[ "$BUILD_WHEELS" == "1" ]]; then
    build_wheelhouse "$VENV_DIR"
  fi

  if [[ "$USE_WHEELHOUSE" == "1" ]]; then
    [[ -d "$WHEELHOUSE_DIR" ]] || die "Wheelhouse directory not found: $WHEELHOUSE_DIR"
  fi

  # Install core deps first.
  pip_install "$VENV_DIR" "$REPO_ROOT/requirements.txt"

  # Install lightweight test/dev deps via pyproject extras to avoid pulling in
  # heavy / potentially conflicting dependency sets.
  if [[ "$INSTALL_TEST_DEPS" == "1" ]]; then
    pip_install_extras "$VENV_DIR" "testing"
  else
    log "Skipping test extras (INSTALL_TEST_DEPS=$INSTALL_TEST_DEPS)"
  fi

  # Install curated base installer requirements if present.
  pip_install "$VENV_DIR" "$REPO_ROOT/install/requirements_base.txt"

  pip_install_editable_package "$VENV_DIR"

  # Optional: install the heavy test stack into an isolated venv.
  if [[ "$INSTALL_HEAVY_TEST_DEPS" == "1" ]]; then
    log "Installing heavy test deps into $TEST_VENV_DIR"
    create_or_update_venv "$TEST_VENV_DIR"
    pip_install "$TEST_VENV_DIR" "$REPO_ROOT/test/requirements.txt"

    # Ensure previous runs didn't leave the project installed in this venv.
    # (Its dependency metadata intentionally differs from the heavy test stack.)
    "$TEST_VENV_DIR/bin/python" -m pip uninstall -y ipfs-accelerate-py ipfs_accelerate_py >/dev/null 2>&1 || true

    # Do NOT install the project package into the heavy test venv.
    # The heavy test stack (e.g. selenium) can legitimately conflict with the
    # project's runtime pins (notably urllib3<2 for ipfshttpclient).
    # Keeping this venv as "tools/tests only" avoids pip check conflicts.
    pip_check "$TEST_VENV_DIR"
  else
    log "Skipping heavy test deps (INSTALL_HEAVY_TEST_DEPS=$INSTALL_HEAVY_TEST_DEPS)"
  fi

  # Install userland tools (no sudo) into ./bin when missing.
  install_gh_to_repo_bin_if_missing
  install_ipfs_kubo_to_repo_bin_if_missing
  install_jq_to_repo_bin_if_missing
  install_yq_to_repo_bin_if_missing

  pip_check "$VENV_DIR"

  log "Done. Open a new shell (or 'source ~/.bashrc') to pick up $BIN_DIR on PATH."
  log "Python: $VENV_DIR/bin/python"
  log "Run tests: $VENV_DIR/bin/python -m pytest"
  if [[ "$INSTALL_HEAVY_TEST_DEPS" == "1" ]]; then
    log "Heavy test env: $TEST_VENV_DIR/bin/python -m pytest (best for selenium/e2e style tests)"
  fi
}

main "$@"
