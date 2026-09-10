"""Project native fleet observations to a real QuackLake DuckLake catalog.

This client owns only an in-memory DuckDB connection and an observational lake
table. It never opens a taskboard database or grants completion authority.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .fleet_history import VIEW_SCHEMA as VIEW_SCHEMA
from .fleet_history import project_view, read_native_view

CONFIG_SCHEMA = "ipfs_accelerate_py/quacklake-catalog-config@1"


def _literal(value: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("invalid SQL literal")
    return "'" + value.replace("'", "''") + "'"


def _private_file(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or not 0 < info.st_size <= 16384):
            raise ValueError("credential file must be private, owned, regular and bounded")
        with os.fdopen(fd, "r", closefd=False) as stream:
            return stream.read(16385).strip()
    finally:
        os.close(fd)


@dataclass(frozen=True)
class CatalogConfig:
    endpoint: str
    data_path: str
    jwt_file: Path
    storage_credentials_file: Path

    @classmethod
    def load(cls, path: Path) -> CatalogConfig:
        data = json.loads(path.read_text())
        if set(data) != {"schema", "endpoint", "data_path", "jwt_file", "storage_credentials_file"} or data["schema"] != CONFIG_SCHEMA:
            raise ValueError("closed QuackLake catalog configuration required")
        endpoint = data["endpoint"]
        if not isinstance(endpoint, str) or not re.fullmatch(r"quack:[A-Za-z0-9][A-Za-z0-9.-]*:443", endpoint):
            raise ValueError("production QuackLake endpoint must be quack:hostname:443")
        parsed = urlsplit(data["data_path"])
        if (parsed.scheme != "r2" or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", parsed.netloc)
                or not re.fullmatch(r"/catalogs/[A-Za-z0-9_-]+/", parsed.path)
                or parsed.query or parsed.fragment):
            raise ValueError("catalog-assigned R2 data path required")
        paths = [Path(data[key]) for key in ("jwt_file", "storage_credentials_file")]
        if not all(item.is_absolute() for item in paths):
            raise ValueError("credential file paths must be absolute")
        return cls(endpoint, data["data_path"], *paths)

    def open(self, *, connect: Any = None) -> Any:
        token = _private_file(self.jwt_file)
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", token):
            raise ValueError("catalog JWT required")
        storage = json.loads(_private_file(self.storage_credentials_file))
        required = {"access_key_id", "secret_access_key", "endpoint"}
        if not required <= set(storage) or set(storage) - required - {"session_token"}:
            raise ValueError("closed R2 credential configuration required")
        if not re.fullmatch(r"[A-Za-z0-9.-]+\.r2\.cloudflarestorage\.com", storage["endpoint"]):
            raise ValueError("R2 S3 endpoint required")
        if any(not isinstance(value, str) or not value for value in storage.values()):
            raise ValueError("nonempty R2 credentials required")
        if connect is None:
            import duckdb
            connect = duckdb.connect
        connection = connect(":memory:")
        try:
            connection.execute("SET threads=1")
            connection.execute("SET memory_limit='256MB'")
            # Use installed, qualified extensions. Never FORCE INSTALL into a
            # shared cache while native owners are running.
            for name in ("quack", "ducklake", "httpfs"):
                connection.execute("LOAD " + name)
            connection.execute("CREATE SECRET fleet_quacklake_catalog (TYPE quack, TOKEN " + _literal(token) + ", SCOPE " + _literal(self.endpoint) + ")")
            options = ["TYPE s3", "PROVIDER config", "KEY_ID " + _literal(storage["access_key_id"]),
                       "SECRET " + _literal(storage["secret_access_key"]), "ENDPOINT " + _literal(storage["endpoint"]),
                       "URL_STYLE 'path'", "REGION 'auto'", "SCOPE " + _literal(self.data_path)]
            if storage.get("session_token"):
                options.append("SESSION_TOKEN " + _literal(storage["session_token"]))
            connection.execute("CREATE SECRET fleet_quacklake_r2 (" + ", ".join(options) + ")")
            connection.execute("ATTACH " + _literal("ducklake:" + self.endpoint) + " AS fleet_lake (DATA_PATH " + _literal(self.data_path) + ")")
            return connection
        except BaseException:
            connection.close()
            raise



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    try:
        config = CatalogConfig.load(args.config)
        if args.check_config:
            print(json.dumps({"status": "configured", "endpoint": config.endpoint, "data_path": config.data_path, "remote_connection_verified": False}))
            return 0
        view = read_native_view(args.deployment, args.inventory)
        connection = config.open()
        try:
            count = project_view(connection, view)
        finally:
            connection.close()
        print(json.dumps({"status": "projected", "source_observations": count, "completion_authority": False}))
        return 0
    except Exception as error:
        # DuckDB exceptions can include the submitted SECRET SQL. Never emit
        # exception messages, SQL, tokens, or storage credentials.
        print(json.dumps({"status": "unavailable", "error_type": type(error).__name__, "completion_authority": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
