"""
Mock implementations of IPFS functionality for the MCP server.

This module provides mock implementations that can be used when the actual
ipfs_kit_py dependency is not available.
"""

import asyncio
import json
import logging
import os
import random
import string
import time
from typing import Any, Dict, List, Optional, Union, cast

logger = logging.getLogger(__name__)

def random_cid() -> str:
    """Generate a random CID-like string for mocking purposes."""
    letters = string.ascii_lowercase + string.digits
    return "Qm" + ''.join(random.choice(letters) for _ in range(44))

class MockIPFSClient:
    """Mock implementation of IPFS client for testing and development."""
    
    def __init__(self):
        """Initialize the mock IPFS client."""
        self.files = {}
        self.pins = {}
        self.mfs = {}
        logger.info("Initialized mock IPFS client")
    
    def add_file(self, path: str, wrap_with_directory: bool = False) -> Dict[str, Any]:
        """Mock implementation of adding a file to IPFS."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        file_size = os.path.getsize(path)
        file_name = os.path.basename(path)
        cid = random_cid()
        
        # Store file info
        self.files[cid] = {
            "path": path,
            "size": file_size,
            "name": file_name,
            "wrapped": wrap_with_directory
        }
        
        return {
            "Hash": cid,
            "Size": file_size,
            "Name": file_name
        }
    
    def cat(self, cid: str, offset: int = 0, length: int = -1) -> bytes:
        """Mock implementation of reading a file from IPFS."""
        if cid not in self.files:
            # Return some placeholder content
            return f"Mock content for CID: {cid}".encode('utf-8')
        
        # Try to read the actual file if available
        try:
            path = self.files[cid]["path"]
            with open(path, "rb") as f:
                if offset > 0:
                    f.seek(offset)
                if length > 0:
                    return f.read(length)
                else:
                    return f.read()
        except Exception:
            return f"Mock content for CID: {cid}".encode('utf-8')
    
    def ls(self, cid: str) -> Dict[str, Any]:
        """Mock implementation of listing directory contents in IPFS."""
        # Create a mock directory listing
        mock_entries = []
        for i in range(3):
            entry_cid = random_cid()
            entry_type = 0  # 0 for file, 1 for directory
            if i == 0:
                entry_type = 1  # Make the first one a directory
            
            mock_entries.append({
                "Name": f"item_{i+1}",
                "Type": entry_type,
                "Size": random.randint(100, 10000),
                "Hash": entry_cid
            })
        
        return {
            "Objects": [
                {
                    "Hash": cid,
                    "Links": mock_entries
                }
            ]
        }
    
    def files_mkdir(self, path: str, parents: bool = False) -> None:
        """Mock implementation of creating a directory in the IPFS MFS."""
        self.mfs[path] = {
            "type": "directory",
            "cid": random_cid(),
            "size": 0,
            "created": time.time()
        }
    
    def files_stat(self, path: str) -> Dict[str, Any]:
        """Mock implementation of getting file stats in the IPFS MFS."""
        if path.startswith("/ipfs/"):
            cid = path[6:]  # Strip "/ipfs/" prefix
            return {
                "Hash": cid,
                "Size": 1024,
                "CumulativeSize": 1024,
                "Blocks": 1,
                "Type": "file"
            }
        
        if path not in self.mfs:
            self.mfs[path] = {
                "type": "file",
                "cid": random_cid(),
                "size": 1024,
                "created": time.time()
            }
        
        entry = self.mfs[path]
        return {
            "Hash": entry["cid"],
            "Size": entry["size"],
            "CumulativeSize": entry["size"],
            "Blocks": 1,
            "Type": entry["type"]
        }
    
    def files_write(self, path: str, data: bytes, create: bool = True, truncate: bool = True) -> None:
        """Mock implementation of writing to a file in the IPFS MFS."""
        self.mfs[path] = {
            "type": "file",
            "cid": random_cid(),
            "size": len(data),
            "created": time.time(),
            "data": data
        }
    
    def files_read(self, path: str, offset: int = 0, count: int = -1) -> bytes:
        """Mock implementation of reading a file from the IPFS MFS."""
        if path not in self.mfs:
            raise FileNotFoundError(f"File not found in MFS: {path}")
        
        if self.mfs[path]["type"] != "file":
            raise ValueError(f"Not a file: {path}")
        
        if "data" in self.mfs[path]:
            data = self.mfs[path]["data"]
        else:
            data = f"Mock content for MFS file: {path}".encode('utf-8')
        
        if offset > 0:
            data = data[offset:]
        
        if count > 0:
            data = data[:count]
        
        return data
    
    def pin_add(self, cid: str, recursive: bool = True) -> Dict[str, Any]:
        """Mock implementation of pinning content in IPFS."""
        self.pins[cid] = {
            "type": "recursive" if recursive else "direct",
            "pinned_at": time.time()
        }
        
        return {
            "Pins": [cid]
        }
    
    def pin_ls(self, cid: str = None) -> Dict[str, Any]:
        """Mock implementation of listing pins in IPFS."""
        if cid is not None:
            if cid not in self.pins:
                return {"Keys": {}}
            
            return {
                "Keys": {
                    cid: {
                        "Type": self.pins[cid]["type"]
                    }
                }
            }
        
        result = {"Keys": {}}
        for pin_cid, pin_info in self.pins.items():
            result["Keys"][pin_cid] = {
                "Type": pin_info["type"]
            }
        
        return result
    
    def pin_rm(self, cid: str, recursive: bool = True) -> Dict[str, Any]:
        """Mock implementation of unpinning content in IPFS."""
        if cid in self.pins:
            del self.pins[cid]
        
        return {
            "Pins": [cid]
        }
    
    def id(self) -> Dict[str, Any]:
        """Mock implementation of getting node identity information."""
        return {
            "ID": "Qm" + ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(44)),
            "PublicKey": "",
            "Addresses": [
                "/ip4/127.0.0.1/tcp/4001",
                "/ip4/192.168.1.100/tcp/4001"
            ],
            "AgentVersion": "ipfs-kit-py/mock",
            "ProtocolVersion": "ipfs/0.1.0"
        }
    
    def swarm_peers(self) -> Dict[str, Any]:
        """Mock implementation of listing connected peers."""
        peers = []
        for i in range(3):
            peer_id = "Qm" + ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(44))
            peers.append({
                "Peer": peer_id,
                "Addr": f"/ip4/192.168.1.{random.randint(1, 254)}/tcp/4001",
                "Latency": f"{random.randint(10, 500)}ms"
            })
        
        return {
            "Peers": peers
        }
    
    def swarm_connect(self, addr: str) -> Dict[str, Any]:
        """Mock implementation of connecting to a peer."""
        return {
            "Strings": [f"Connection success: {addr}"]
        }
    
    def pubsub_pub(self, topic: str, message: Union[str, bytes]) -> None:
        """Mock implementation of publishing to a pubsub topic."""
        pass
    
    def dht_findpeer(self, peer_id: str) -> Dict[str, Any]:
        """Mock implementation of finding a peer in the DHT."""
        return {
            "Responses": [
                {
                    "ID": peer_id,
                    "Addrs": [
                        f"/ip4/192.168.1.{random.randint(1, 254)}/tcp/4001",
                        f"/ip4/172.16.{random.randint(1, 254)}.{random.randint(1, 254)}/tcp/4001"
                    ]
                }
            ]
        }
    
    def dht_findprovs(self, cid: str, num_providers: int = 20) -> Dict[str, Any]:
        """Mock implementation of finding providers for a CID in the DHT."""
        responses = []
        for i in range(min(num_providers, 5)):
            peer_id = "Qm" + ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(44))
            responses.append({
                "ID": peer_id,
                "Addrs": [
                    f"/ip4/192.168.1.{random.randint(1, 254)}/tcp/4001",
                    f"/ip4/172.16.{random.randint(1, 254)}.{random.randint(1, 254)}/tcp/4001"
                ]
            })
        
        return {
            "Responses": responses
        }
    
    def version(self) -> Dict[str, Any]:
        """Get the IPFS version."""
        return {
            "Version": "ipfs-kit-py/mock",
            "Commit": "",
            "Repo": "7",
            "System": "ipfs-kit-py/mock",
            "Golang": "go-mock/1.0.0"
        }
