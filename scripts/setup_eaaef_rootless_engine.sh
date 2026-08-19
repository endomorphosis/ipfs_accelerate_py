#!/bin/sh
# Install and start a user-owned rootless Docker engine for EAAEF-183.
# Does not replace the host rootful daemon and does not mount /var/run/docker.sock.
set -eu

uid="$(id -u)"
runtime_dir="${XDG_RUNTIME_DIR:-/run/user/${uid}}"
rootless_host="unix://${runtime_dir}/docker.sock"

if [ "${uid}" = "0" ]; then
  echo "refusing to install rootless Docker as uid 0" >&2
  exit 1
fi

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
  dockerd-rootless-setuptool.sh install --skip-iptables || true
else
  echo "dockerd-rootless-setuptool.sh is missing after package install" >&2
  exit 1
fi

systemctl --user enable --now docker.service || true
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
