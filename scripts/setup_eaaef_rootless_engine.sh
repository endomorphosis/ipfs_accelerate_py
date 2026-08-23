#!/bin/sh
# Install packages as root if needed, then start a *user-owned* rootless
# Docker engine for EAAEF-183. Never start rootless dockerd as uid 0.
set -eu

script="$(readlink -f "$0")"

install_packages() {
  need_apt=""
  command -v newuidmap >/dev/null 2>&1 || need_apt="${need_apt} uidmap"
  command -v slirp4netns >/dev/null 2>&1 || need_apt="${need_apt} slirp4netns"
  command -v fuse-overlayfs >/dev/null 2>&1 || need_apt="${need_apt} fuse-overlayfs"
  command -v dockerd-rootless.sh >/dev/null 2>&1 || need_apt="${need_apt} docker-ce-rootless-extras"
  [ -z "${need_apt}" ] && return 0
  echo "installing rootless engine packages:${need_apt}"
  apt-get update -y
  apt-get install -y ${need_apt}
}

if [ "$(id -u)" = "0" ]; then
  install_packages
  target_user="${SUDO_USER:-}"
  if [ -z "${target_user}" ] || [ "${target_user}" = "root" ]; then
    echo "packages are installed. re-run this script as the unprivileged account, not root:" >&2
    echo "  sudo -u barberb -H ${script}" >&2
    echo "or:" >&2
    echo "  exit the root shell, then ${script}" >&2
    exit 1
  fi
  target_uid="$(id -u "${target_user}")"
  target_runtime="/run/user/${target_uid}"
  echo "dropping privileges to ${target_user} for rootless daemon install"
  exec runuser -u "${target_user}" -- env \
    HOME="$(getent passwd "${target_user}" | cut -d: -f6)" \
    USER="${target_user}" \
    LOGNAME="${target_user}" \
    XDG_RUNTIME_DIR="${target_runtime}" \
    "${script}"
fi

uid="$(id -u)"
runtime_dir="${XDG_RUNTIME_DIR:-/run/user/${uid}}"
rootless_host="unix://${runtime_dir}/docker.sock"

need_apt=""
command -v newuidmap >/dev/null 2>&1 || need_apt="${need_apt} uidmap"
command -v slirp4netns >/dev/null 2>&1 || need_apt="${need_apt} slirp4netns"
command -v fuse-overlayfs >/dev/null 2>&1 || need_apt="${need_apt} fuse-overlayfs"
command -v dockerd-rootless.sh >/dev/null 2>&1 || need_apt="${need_apt} docker-ce-rootless-extras"
if [ -n "${need_apt}" ]; then
  echo "installing rootless engine packages:${need_apt}"
  sudo apt-get update -y
  sudo apt-get install -y ${need_apt}
fi

if ! grep -q "^$(id -un):" /etc/subuid || ! grep -q "^$(id -un):" /etc/subgid; then
  echo "host is missing /etc/subuid or /etc/subgid entries for $(id -un)" >&2
  exit 1
fi

export XDG_RUNTIME_DIR="${runtime_dir}"
export PATH="/usr/bin:${PATH}"
if command -v dockerd-rootless-setuptool.sh >/dev/null 2>&1; then
  # --force: this host already has a rootful daemon on /var/run/docker.sock.
  # EAAEF uses the user socket, not that socket, so both may coexist.
  dockerd-rootless-setuptool.sh --force --skip-iptables install
else
  echo "dockerd-rootless-setuptool.sh is missing after package install" >&2
  exit 1
fi

systemctl --user enable --now docker.service
loginctl enable-linger "$(id -un)" >/dev/null 2>&1 || true

if ! docker -H "${rootless_host}" info >/dev/null 2>&1; then
  echo "rootless dockerd did not become ready at ${rootless_host}" >&2
  docker -H "${rootless_host}" info >&2 || true
  exit 1
fi

security="$(docker -H "${rootless_host}" info --format '{{json .SecurityOptions}}')"
case "${security}" in
  *rootless*) ;;
  *)
    echo "rootless dockerd is running but SecurityOptions lack rootless: ${security}" >&2
    exit 1
    ;;
esac

echo "EAAEF_DOCKER_HOST=${rootless_host}"
echo "verified_rootless=true"
