#!/usr/bin/env python3
"""Generate a self-signed TLS certificate (first-run helper for systemd).

Designed for LAN/IP-based HTTPS where you don't yet have a real domain.

- If cert/key already exist, it's a no-op (unless --force).
- Generates a self-signed cert with SANs including:
  - IP:<detected LAN IP> (or --ip)
  - DNS:<hostname> and DNS:<fqdn>

This script intentionally shells out to `openssl` because it's ubiquitous on
Linux servers and avoids adding Python crypto deps.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import socket
import subprocess
import sys
from typing import Optional


def _detect_lan_ip() -> Optional[str]:
    """Best-effort: pick the source IP used to reach the internet.

    This does not send packets, but it does require the routing table.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # No traffic is sent for UDP connect; used to select a route.
            s.connect(("1.1.1.1", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        return None
    return None


def _hostname_sans() -> list[str]:
    sans: list[str] = []
    try:
        hn = socket.gethostname().strip()
        if hn:
            sans.append(f"DNS:{hn}")
    except Exception:
        pass
    try:
        fqdn = socket.getfqdn().strip()
        if fqdn and fqdn not in {socket.gethostname().strip(), "localhost"}:
            sans.append(f"DNS:{fqdn}")
    except Exception:
        pass
    # Always include localhost for convenience.
    sans.append("DNS:localhost")
    return sans


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")


def _read_env_var(path: pathlib.Path, key: str) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""
    return ""


def _is_auto_managed_path(cert_path: pathlib.Path, key_path: pathlib.Path) -> bool:
    # Only auto-regenerate certs if they live under .../state/tls/.
    # This avoids clobbering operator-managed certs like Let's Encrypt.
    def _managed(p: pathlib.Path) -> bool:
        parts = [x.lower() for x in p.parts]
        try:
            i = parts.index("state")
            return i + 1 < len(parts) and parts[i + 1] == "tls"
        except ValueError:
            return False

    return _managed(cert_path) and _managed(key_path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate self-signed TLS cert/key if missing")
    parser.add_argument("--certfile", required=True, help="Path to write the certificate (PEM)")
    parser.add_argument("--keyfile", required=True, help="Path to write the private key (PEM)")
    parser.add_argument("--ip", default="", help="LAN IP to include in SAN (default: auto-detect)")
    parser.add_argument(
        "--write-env-file",
        default="",
        help="Optional path to write MCP_LAN_IP=... (updated when value changes)",
    )
    parser.add_argument("--days", type=int, default=365, help="Validity period in days")
    parser.add_argument("--cn", default="ipfs-accelerate-mcpplusplus", help="Certificate common name")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")

    args = parser.parse_args(argv)

    cert_path = pathlib.Path(args.certfile).expanduser()
    key_path = pathlib.Path(args.keyfile).expanduser()

    override_ip = (args.ip or "").strip()
    detected_ip = _detect_lan_ip() or ""
    ip = override_ip or detected_ip

    env_path = pathlib.Path(args.write_env_file).expanduser() if str(args.write_env_file or "").strip() else None
    prev_env_ip = ""
    if env_path is not None:
        prev_env_ip = _read_env_var(env_path, "MCP_LAN_IP")
        if ip:
            if args.force or prev_env_ip != ip:
                env_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_env = env_path.with_suffix(env_path.suffix + ".tmp")
                tmp_env.write_text(f"MCP_LAN_IP={ip}\n", encoding="utf-8")
                tmp_env.replace(env_path)

    cert_exists = cert_path.exists()
    key_exists = key_path.exists()
    ip_changed = bool(prev_env_ip and ip and prev_env_ip != ip)
    should_regen_for_ip_change = ip_changed and _is_auto_managed_path(cert_path, key_path)

    if not args.force and cert_exists and key_exists and not should_regen_for_ip_change:
        return 0

    openssl = shutil.which("openssl")
    if not openssl:
        print("ERROR: openssl not found; cannot generate self-signed cert", file=sys.stderr)
        return 2

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    san_parts = []
    if ip:
        san_parts.append(f"IP:{ip}")
    san_parts.extend(_hostname_sans())

    san = ",".join(san_parts)

    # Write into temp files first to avoid partial writes.
    tmp_key = key_path.with_suffix(key_path.suffix + ".tmp")
    tmp_crt = cert_path.with_suffix(cert_path.suffix + ".tmp")

    for p in (tmp_key, tmp_crt):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    # Ensure target overwrite allowed.
    if args.force:
        for p in (key_path, cert_path):
            if p.exists():
                p.unlink()
    elif should_regen_for_ip_change:
        for p in (key_path, cert_path):
            if p.exists():
                p.unlink()

    cmd = [
        openssl,
        "req",
        "-x509",
        "-nodes",
        "-newkey",
        "rsa:2048",
        "-days",
        str(int(args.days)),
        "-keyout",
        str(tmp_key),
        "-out",
        str(tmp_crt),
        "-subj",
        f"/CN={args.cn}",
        "-addext",
        f"subjectAltName={san}",
    ]

    _run(cmd)

    # Permissions: key should not be world-readable.
    try:
        os.chmod(tmp_key, 0o640)
    except Exception:
        pass

    tmp_key.replace(key_path)
    tmp_crt.replace(cert_path)

    print(f"Generated self-signed TLS cert: {cert_path}")
    print(f"Generated self-signed TLS key:  {key_path}")
    if ip:
        print(f"Included SAN IP: {ip}")
    # Cert already exists; still successful (env file may have been written above).

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
