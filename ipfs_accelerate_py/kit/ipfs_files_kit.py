"""
IPFS Files Kit Module

This module provides a unified interface for IPFS file operations by wrapping
the external ipfs_kit_py package. It follows the kit pattern established for
other modules (github_kit, docker_kit, etc.) and provides core functionality
that is then exposed through both CLI and MCP tools.

Architecture:
    External ipfs_kit_py package (git submodule)
        ↓
    ipfs_files_kit.py (this module - wraps external package)
        ↓
    ├─ unified_cli.py (CLI wrapper)
    └─ mcp/unified_tools.py (MCP wrapper)

Key Features:
- Add files to IPFS and get CIDs
- Retrieve files from IPFS by CID
- Pin/unpin content for persistence
- List and manage IPFS files
- CID validation and verification
- Graceful fallback when ipfs_kit_py unavailable

Usage:
    from ipfs_accelerate_py.kit.ipfs_files_kit import IPFSFilesKit, IPFSFilesConfig
    
    # Initialize
    config = IPFSFilesConfig()
    kit = IPFSFilesKit(config)
    
    # Add file
    result = kit.add_file("/path/to/file.txt")
    print(f"CID: {result.data['cid']}")
    
    # Get file
    result = kit.get_file("Qm...", "/path/to/output.txt")
    
    # List files
    result = kit.list_files()
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union

# Configure logging
logger = logging.getLogger("ipfs_accelerate.kit.ipfs_files")


@dataclass
class IPFSFilesConfig:
    """Configuration for IPFS files operations."""
    
    # Whether to enable ipfs_kit_py integration
    enable_ipfs_kit: bool = True
    
    # Cache directory for local operations
    cache_dir: str = "~/.cache/ipfs_accelerate"
    
    # IPFS gateway URL for fallback
    ipfs_gateway: Optional[str] = None
    
    # Timeout for operations (seconds)
    timeout: int = 30
    
    # Whether to use local IPFS node if available
    use_local_node: bool = True
    
    # Whether to auto-pin added files
    auto_pin: bool = True


@dataclass
class IPFSFileInfo:
    """Information about an IPFS file."""
    
    cid: str
    name: str
    size: int
    type: str = "file"  # file or directory
    pinned: bool = False
    path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class IPFSFileResult:
    """Result from an IPFS file operation."""
    
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class IPFSFilesKit:
    """
    IPFS files operations kit.
    
    Provides unified interface for IPFS file operations by wrapping the external
    ipfs_kit_py package. Includes graceful fallback when the package is unavailable.
    """
    
    def __init__(self, config: Optional[IPFSFilesConfig] = None):
        """
        Initialize IPFS files kit.
        
        Args:
            config: Configuration for IPFS operations
        """
        self.config = config or IPFSFilesConfig()
        self.ipfs_client = None
        self.ipfs_available = False
        
        # Try to initialize ipfs_kit_py client
        if self.config.enable_ipfs_kit:
            self._init_ipfs_client()
    
    def _init_ipfs_client(self):
        """Initialize IPFS client from ipfs_kit_py package."""
        try:
            # Try importing the external ipfs_kit_py package
            import ipfs_kit_py
            
            # Try to get the storage wrapper or client
            try:
                from ipfs_accelerate_py.ipfs_kit_integration import IPFSKitStorage
                self.ipfs_client = IPFSKitStorage(
                    enable_ipfs_kit=True,
                    cache_dir=self.config.cache_dir
                )
                self.ipfs_available = True
                logger.info("IPFS client initialized using ipfs_kit_integration")
            except ImportError:
                # Try direct import
                try:
                    from ipfs_kit_py import IPFSApi
                    self.ipfs_client = IPFSApi()
                    self.ipfs_available = True
                    logger.info("IPFS client initialized using IPFSApi")
                except (ImportError, AttributeError):
                    logger.warning("Could not initialize ipfs_kit_py client")
                    
        except ImportError as e:
            logger.warning(f"ipfs_kit_py not available: {e}. Using fallback mode.")
    
    def add_file(
        self,
        path: str,
        pin: Optional[bool] = None
    ) -> IPFSFileResult:
        """
        Add a file to IPFS.
        
        Args:
            path: Path to file to add
            pin: Whether to pin the file (default: from config)
            
        Returns:
            IPFSFileResult with CID and file info
        """
        if pin is None:
            pin = self.config.auto_pin
        
        path = os.path.expanduser(path)
        
        if not os.path.exists(path):
            return IPFSFileResult(
                success=False,
                message=f"File not found: {path}",
                error="FileNotFoundError"
            )
        
        try:
            # Try using ipfs_kit_py if available
            if self.ipfs_available and self.ipfs_client:
                try:
                    # Try store method if available (IPFSKitStorage)
                    if hasattr(self.ipfs_client, 'store'):
                        cid = self.ipfs_client.store(
                            open(path, 'rb').read(),
                            os.path.basename(path)
                        )
                    # Try add method if available
                    elif hasattr(self.ipfs_client, 'add'):
                        result = self.ipfs_client.add(path)
                        cid = result.get('Hash') or result.get('cid')
                    else:
                        raise AttributeError("No suitable add method found")
                    
                    file_info = IPFSFileInfo(
                        cid=cid,
                        name=os.path.basename(path),
                        size=os.path.getsize(path),
                        pinned=pin,
                        path=path
                    )
                    
                    return IPFSFileResult(
                        success=True,
                        message=f"File added to IPFS: {cid}",
                        data=file_info.to_dict()
                    )
                    
                except Exception as e:
                    logger.warning(f"ipfs_kit_py add failed, trying fallback: {e}")
            
            # Fallback: try using IPFS CLI if available
            try:
                cmd = ['ipfs', 'add', '-Q', path]
                if not pin:
                    cmd.insert(2, '--pin=false')
                    
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    cid = result.stdout.strip()
                    file_info = IPFSFileInfo(
                        cid=cid,
                        name=os.path.basename(path),
                        size=os.path.getsize(path),
                        pinned=pin,
                        path=path
                    )
                    
                    return IPFSFileResult(
                        success=True,
                        message=f"File added to IPFS (CLI): {cid}",
                        data=file_info.to_dict()
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, cmd, result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return IPFSFileResult(
                    success=False,
                    message="IPFS not available (no ipfs_kit_py or CLI)",
                    error=str(e)
                )
                
        except Exception as e:
            logger.error(f"Error adding file to IPFS: {e}")
            return IPFSFileResult(
                success=False,
                message=f"Error adding file: {str(e)}",
                error=type(e).__name__
            )
    
    def get_file(
        self,
        cid: str,
        output_path: str
    ) -> IPFSFileResult:
        """
        Get a file from IPFS by CID.
        
        Args:
            cid: IPFS CID of the file
            output_path: Where to save the file
            
        Returns:
            IPFSFileResult with file info
        """
        output_path = os.path.expanduser(output_path)
        
        try:
            # Try using ipfs_kit_py if available
            if self.ipfs_available and self.ipfs_client:
                try:
                    if hasattr(self.ipfs_client, 'retrieve'):
                        data = self.ipfs_client.retrieve(cid)
                        with open(output_path, 'wb') as f:
                            f.write(data)
                    elif hasattr(self.ipfs_client, 'get'):
                        data = self.ipfs_client.get(cid)
                        with open(output_path, 'wb') as f:
                            f.write(data)
                    else:
                        raise AttributeError("No suitable get method found")
                    
                    file_info = IPFSFileInfo(
                        cid=cid,
                        name=os.path.basename(output_path),
                        size=os.path.getsize(output_path),
                        path=output_path
                    )
                    
                    return IPFSFileResult(
                        success=True,
                        message=f"File retrieved from IPFS: {cid}",
                        data=file_info.to_dict()
                    )
                    
                except Exception as e:
                    logger.warning(f"ipfs_kit_py get failed, trying fallback: {e}")
            
            # Fallback: try using IPFS CLI
            try:
                result = subprocess.run(
                    ['ipfs', 'get', cid, '-o', output_path],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    file_info = IPFSFileInfo(
                        cid=cid,
                        name=os.path.basename(output_path),
                        size=os.path.getsize(output_path),
                        path=output_path
                    )
                    
                    return IPFSFileResult(
                        success=True,
                        message=f"File retrieved from IPFS (CLI): {cid}",
                        data=file_info.to_dict()
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, ['ipfs', 'get'], result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return IPFSFileResult(
                    success=False,
                    message="IPFS not available (no ipfs_kit_py or CLI)",
                    error=str(e)
                )
                
        except Exception as e:
            logger.error(f"Error getting file from IPFS: {e}")
            return IPFSFileResult(
                success=False,
                message=f"Error getting file: {str(e)}",
                error=type(e).__name__
            )
    
    def cat_file(self, cid: str) -> IPFSFileResult:
        """
        Read file content from IPFS by CID.
        
        Args:
            cid: IPFS CID of the file
            
        Returns:
            IPFSFileResult with file content in data['content']
        """
        try:
            # Try using ipfs_kit_py if available
            if self.ipfs_available and self.ipfs_client:
                try:
                    if hasattr(self.ipfs_client, 'retrieve'):
                        data = self.ipfs_client.retrieve(cid)
                    elif hasattr(self.ipfs_client, 'cat'):
                        data = self.ipfs_client.cat(cid)
                    else:
                        raise AttributeError("No suitable cat method found")
                    
                    # Try to decode as text, otherwise return as bytes
                    try:
                        content = data.decode('utf-8')
                    except (UnicodeDecodeError, AttributeError):
                        content = data if isinstance(data, str) else str(data)
                    
                    return IPFSFileResult(
                        success=True,
                        message=f"File content retrieved: {cid}",
                        data={
                            'cid': cid,
                            'content': content,
                            'size': len(data) if isinstance(data, (bytes, str)) else 0
                        }
                    )
                    
                except Exception as e:
                    logger.warning(f"ipfs_kit_py cat failed, trying fallback: {e}")
            
            # Fallback: try using IPFS CLI
            try:
                result = subprocess.run(
                    ['ipfs', 'cat', cid],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    return IPFSFileResult(
                        success=True,
                        message=f"File content retrieved (CLI): {cid}",
                        data={
                            'cid': cid,
                            'content': result.stdout,
                            'size': len(result.stdout)
                        }
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, ['ipfs', 'cat'], result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return IPFSFileResult(
                    success=False,
                    message="IPFS not available (no ipfs_kit_py or CLI)",
                    error=str(e)
                )
                
        except Exception as e:
            logger.error(f"Error reading file from IPFS: {e}")
            return IPFSFileResult(
                success=False,
                message=f"Error reading file: {str(e)}",
                error=type(e).__name__
            )
    
    def pin_file(self, cid: str) -> IPFSFileResult:
        """
        Pin a file in IPFS (keep it cached locally).
        
        Args:
            cid: IPFS CID to pin
            
        Returns:
            IPFSFileResult with pin status
        """
        try:
            # Try using IPFS CLI for pinning
            try:
                result = subprocess.run(
                    ['ipfs', 'pin', 'add', cid],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    return IPFSFileResult(
                        success=True,
                        message=f"File pinned: {cid}",
                        data={'cid': cid, 'pinned': True}
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, ['ipfs', 'pin', 'add'], result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return IPFSFileResult(
                    success=False,
                    message="IPFS not available for pinning",
                    error=str(e)
                )
                
        except Exception as e:
            logger.error(f"Error pinning file: {e}")
            return IPFSFileResult(
                success=False,
                message=f"Error pinning file: {str(e)}",
                error=type(e).__name__
            )
    
    def unpin_file(self, cid: str) -> IPFSFileResult:
        """
        Unpin a file in IPFS (allow garbage collection).
        
        Args:
            cid: IPFS CID to unpin
            
        Returns:
            IPFSFileResult with unpin status
        """
        try:
            # Try using IPFS CLI for unpinning
            try:
                result = subprocess.run(
                    ['ipfs', 'pin', 'rm', cid],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    return IPFSFileResult(
                        success=True,
                        message=f"File unpinned: {cid}",
                        data={'cid': cid, 'pinned': False}
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, ['ipfs', 'pin', 'rm'], result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return IPFSFileResult(
                    success=False,
                    message="IPFS not available for unpinning",
                    error=str(e)
                )
                
        except Exception as e:
            logger.error(f"Error unpinning file: {e}")
            return IPFSFileResult(
                success=False,
                message=f"Error unpinning file: {str(e)}",
                error=type(e).__name__
            )
    
    def list_files(self, path: str = "/") -> IPFSFileResult:
        """
        List files in IPFS.
        
        Args:
            path: IPFS path to list (default: root)
            
        Returns:
            IPFSFileResult with list of files
        """
        try:
            # Try using IPFS CLI
            try:
                result = subprocess.run(
                    ['ipfs', 'files', 'ls', '-l', path],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    files = []
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            # Parse IPFS files ls output
                            parts = line.split()
                            if len(parts) >= 3:
                                files.append({
                                    'name': parts[-1],
                                    'size': int(parts[-2]) if parts[-2].isdigit() else 0,
                                    'cid': parts[0] if len(parts) > 3 else 'unknown'
                                })
                    
                    return IPFSFileResult(
                        success=True,
                        message=f"Listed files in {path}",
                        data={'path': path, 'files': files, 'count': len(files)}
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, ['ipfs', 'files', 'ls'], result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return IPFSFileResult(
                    success=False,
                    message="IPFS not available for listing",
                    error=str(e)
                )
                
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return IPFSFileResult(
                success=False,
                message=f"Error listing files: {str(e)}",
                error=type(e).__name__
            )
    
    def validate_cid(self, cid: str) -> IPFSFileResult:
        """
        Validate an IPFS CID format.
        
        Args:
            cid: CID to validate
            
        Returns:
            IPFSFileResult with validation result
        """
        try:
            # Basic CID validation
            if not cid or not isinstance(cid, str):
                return IPFSFileResult(
                    success=False,
                    message="Invalid CID: must be a non-empty string",
                    data={'cid': cid, 'valid': False}
                )
            
            # Check for common CID prefixes
            valid_prefixes = ('Qm', 'bafy', 'bafk', 'bafyre')
            is_valid = cid.startswith(valid_prefixes) and len(cid) >= 46
            
            # Try using IPFS CLI for validation if available
            if is_valid:
                try:
                    result = subprocess.run(
                        ['ipfs', 'cid', 'format', cid],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    is_valid = result.returncode == 0
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass  # Keep basic validation result
            
            return IPFSFileResult(
                success=is_valid,
                message=f"CID {'is valid' if is_valid else 'is invalid'}",
                data={'cid': cid, 'valid': is_valid}
            )
            
        except Exception as e:
            logger.error(f"Error validating CID: {e}")
            return IPFSFileResult(
                success=False,
                message=f"Error validating CID: {str(e)}",
                error=type(e).__name__
            )


# Singleton instance getter
_ipfs_files_kit_instance = None

def get_ipfs_files_kit(config: Optional[IPFSFilesConfig] = None) -> IPFSFilesKit:
    """
    Get the singleton IPFSFilesKit instance.
    
    Args:
        config: Optional configuration (used only for first call)
        
    Returns:
        IPFSFilesKit instance
    """
    global _ipfs_files_kit_instance
    if _ipfs_files_kit_instance is None:
        _ipfs_files_kit_instance = IPFSFilesKit(config)
    return _ipfs_files_kit_instance
