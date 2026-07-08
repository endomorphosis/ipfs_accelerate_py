# Zero-Touch Installer for ipfs_accelerate_py Cache Infrastructure (Windows)
# Supports: Windows 10/11 (x64, ARM64)
#
# Usage:
#   iwr -useb https://raw.githubusercontent.com/endomorphosis/ipfs_accelerate_py/main/install/install.ps1 | iex
#   or
#   .\install.ps1 [-Profile minimal|standard|full|cli] [-Silent] [-NoCliTools]

param(
    [string]$Profile = "standard",
    [string]$CacheDir = "$env:USERPROFILE\.cache\ipfs_accelerate_py",
    [switch]$Silent = $false,
    [switch]$NoCliTools = $false,
    [switch]$SkipVenv = $false,
    [switch]$Verify = $false,
    [switch]$Verbose = $false,
    [switch]$Help = $false
)

# Configuration
$ErrorActionPreference = "Stop"
$LogFile = "$env:USERPROFILE\.cache\ipfs_accelerate_py\install.log"

# Colors
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    
    # Write to log file
    $logDir = Split-Path $LogFile -Parent
    if (!(Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    Add-Content -Path $LogFile -Value $logMessage
    
    # Print to console if not silent
    if (!$Silent) {
        switch ($Level) {
            "INFO"    { Write-Host "ℹ $Message" -ForegroundColor Cyan }
            "SUCCESS" { Write-Host "✓ $Message" -ForegroundColor Green }
            "WARN"    { Write-Host "⚠ $Message" -ForegroundColor Yellow }
            "ERROR"   { Write-Host "✗ $Message" -ForegroundColor Red }
        }
    }
}

function Exit-WithError {
    param([string]$Message)
    Write-ColorOutput -Message $Message -Level ERROR
    exit 1
}

# Help
if ($Help) {
    @"
Zero-Touch Installer for ipfs_accelerate_py

Usage: .\install.ps1 [OPTIONS]

Options:
  -Profile PROFILE       Installation profile (minimal|standard|full|cli)
  -CacheDir DIR         Cache directory (default: ~\.cache\ipfs_accelerate_py)
  -Silent               Silent installation (no interactive prompts)
  -NoCliTools           Skip CLI tool installation
  -SkipVenv             Skip virtual environment creation
  -Verify               Verify existing installation
  -Verbose              Verbose output
  -Help                 Show this help message

Examples:
  .\install.ps1 -Profile minimal
  .\install.ps1 -Profile standard -CacheDir C:\cache
  .\install.ps1 -Silent -NoCliTools

"@
    exit 0
}

# Banner
if (!$Silent) {
    @"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ipfs_accelerate_py Cache Infrastructure Installer          ║
║   Zero-Touch Multi-Platform Installation (Windows)           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

"@
}

Write-ColorOutput -Message "Starting installation with profile: $Profile" -Level INFO

# Detect architecture
function Get-Architecture {
    $arch = $env:PROCESSOR_ARCHITECTURE
    switch ($arch) {
        "AMD64" { return "x64" }
        "ARM64" { return "arm64" }
        default { Exit-WithError "Unsupported architecture: $arch" }
    }
}

$ARCH = Get-Architecture
Write-ColorOutput -Message "Detected architecture: $ARCH" -Level INFO

# Find Python
function Find-Python {
    $pythonCommands = @("python3.12", "python3.11", "python3.10", "python3.9", "python3", "python", "py")
    
    foreach ($py in $pythonCommands) {
        try {
            $version = & $py --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput -Message "Found Python: $py ($version)" -Level INFO
                return $py
            }
        } catch {
            continue
        }
    }
    
    Exit-WithError "Python 3.8+ not found. Please install Python from python.org"
}

$PythonCmd = Find-Python

# Check if running in virtual environment
function Test-VirtualEnv {
    return $null -ne $env:VIRTUAL_ENV
}

# Create virtual environment
function New-VirtualEnv {
    if ((Test-VirtualEnv) -or $SkipVenv) {
        return
    }
    
    Write-ColorOutput -Message "Creating virtual environment..." -Level INFO
    $venvDir = ".venv"
    
    if (Test-Path $venvDir) {
        Write-ColorOutput -Message "Virtual environment already exists at $venvDir" -Level WARN
        if (!$Silent) {
            $response = Read-Host "Use existing virtual environment? (y/n)"
            if ($response -ne "y") {
                Remove-Item -Recurse -Force $venvDir
                & $PythonCmd -m venv $venvDir
                if ($LASTEXITCODE -ne 0) {
                    Exit-WithError "Failed to create virtual environment"
                }
            }
        }
    } else {
        & $PythonCmd -m venv $venvDir
        if ($LASTEXITCODE -ne 0) {
            Exit-WithError "Failed to create virtual environment"
        }
    }
    
    # Activate venv
    $activateScript = Join-Path $venvDir "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        $script:PythonCmd = "python"
        Write-ColorOutput -Message "Virtual environment created and activated" -Level SUCCESS
    } else {
        Exit-WithError "Failed to find activation script"
    }
}

# Install Python dependencies
function Install-PythonDeps {
    Write-ColorOutput -Message "Installing Python dependencies for profile: $Profile" -Level INFO
    
    # Upgrade pip
    & $PythonCmd -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        Exit-WithError "Failed to upgrade pip"
    }
    
    # Install based on profile
    switch ($Profile) {
        "minimal" {
            & $PythonCmd -m pip install "ipfs_accelerate_py[minimal]"
        }
        "standard" {
            & $PythonCmd -m pip install "ipfs_accelerate_py[cache]"
        }
        "full" {
            & $PythonCmd -m pip install "ipfs_accelerate_py[all]"
        }
        "cli" {
            & $PythonCmd -m pip install "ipfs_accelerate_py[cli]"
        }
        default {
            Exit-WithError "Unknown profile: $Profile"
        }
    }
    
    if ($LASTEXITCODE -ne 0) {
        Exit-WithError "Failed to install $Profile profile"
    }
    
    Write-ColorOutput -Message "Python dependencies installed" -Level SUCCESS
}

# Install CLI tools
function Install-CliTools {
    if ($NoCliTools) {
        Write-ColorOutput -Message "Skipping CLI tools installation" -Level INFO
        return
    }
    
    Write-ColorOutput -Message "Installing CLI tools..." -Level INFO
    
    # Check for winget
    $hasWinget = Get-Command winget -ErrorAction SilentlyContinue
    
    # GitHub CLI
    if (!(Get-Command gh -ErrorAction SilentlyContinue)) {
        Write-ColorOutput -Message "Installing GitHub CLI..." -Level INFO
        if ($hasWinget) {
            winget install --id GitHub.cli --silent
        } else {
            Write-ColorOutput -Message "winget not found. Please install GitHub CLI manually from https://cli.github.com" -Level WARN
        }
    } else {
        Write-ColorOutput -Message "GitHub CLI already installed" -Level INFO
    }
    
    # HuggingFace CLI
    if (!(Get-Command huggingface-cli -ErrorAction SilentlyContinue)) {
        Write-ColorOutput -Message "Installing HuggingFace CLI..." -Level INFO
        & $PythonCmd -m pip install "huggingface-hub[cli]"
    } else {
        Write-ColorOutput -Message "HuggingFace CLI already installed" -Level INFO
    }
    
    # Vast AI CLI
    if (!(Get-Command vastai -ErrorAction SilentlyContinue)) {
        Write-ColorOutput -Message "Installing Vast AI CLI..." -Level INFO
        & $PythonCmd -m pip install vastai
    } else {
        Write-ColorOutput -Message "Vast AI CLI already installed" -Level INFO
    }
}

# Setup cache directory
function New-CacheDir {
    Write-ColorOutput -Message "Setting up cache directory: $CacheDir" -Level INFO
    
    if (!(Test-Path $CacheDir)) {
        New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null
    }
    
    Write-ColorOutput -Message "Cache directory created" -Level SUCCESS
}

# Configure environment
function Set-Environment {
    Write-ColorOutput -Message "Configuring environment variables..." -Level INFO
    
    # Set user environment variables
    [Environment]::SetEnvironmentVariable("IPFS_ACCELERATE_CACHE_DIR", $CacheDir, "User")
    [Environment]::SetEnvironmentVariable("IPFS_ACCELERATE_CACHE_TTL", "3600", "User")
    
    Write-ColorOutput -Message "Environment variables configured" -Level SUCCESS
}

# Verify installation
function Test-Installation {
    Write-ColorOutput -Message "Verifying installation..." -Level INFO
    
    $errors = 0
    
    # Test Python imports
    $imports = @(
        "from ipfs_accelerate_py.common.base_cache import BaseCache",
        "from ipfs_accelerate_py.common.cid_index import CIDIndex",
        "from ipfs_accelerate_py.common.llm_cache import get_global_llm_cache"
    )
    
    foreach ($import in $imports) {
        try {
            & $PythonCmd -c $import 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput -Message "$import - OK" -Level SUCCESS
            } else {
                Write-ColorOutput -Message "$import - FAILED" -Level ERROR
                $errors++
            }
        } catch {
            Write-ColorOutput -Message "$import - FAILED" -Level ERROR
            $errors++
        }
    }
    
    # Test CLI integrations
    if ($Profile -ne "minimal") {
        try {
            & $PythonCmd -c "from ipfs_accelerate_py.cli_integrations import get_all_cli_integrations" 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput -Message "CLI integrations - OK" -Level SUCCESS
            } else {
                Write-ColorOutput -Message "CLI integrations - FAILED" -Level ERROR
                $errors++
            }
        } catch {
            Write-ColorOutput -Message "CLI integrations - FAILED" -Level ERROR
            $errors++
        }
    }
    
    # Test API integrations
    if ($Profile -eq "standard" -or $Profile -eq "full") {
        try {
            & $PythonCmd -c "from ipfs_accelerate_py.api_integrations import get_cached_openai_api" 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput -Message "API integrations - OK" -Level SUCCESS
            } else {
                Write-ColorOutput -Message "API integrations - FAILED" -Level ERROR
                $errors++
            }
        } catch {
            Write-ColorOutput -Message "API integrations - FAILED" -Level ERROR
            $errors++
        }
    }
    
    # Check cache directory
    if (Test-Path $CacheDir) {
        Write-ColorOutput -Message "Cache directory exists - OK" -Level SUCCESS
    } else {
        Write-ColorOutput -Message "Cache directory not found - FAILED" -Level ERROR
        $errors++
    }
    
    if ($errors -eq 0) {
        Write-ColorOutput -Message "All verification checks passed" -Level SUCCESS
        return $true
    } else {
        Write-ColorOutput -Message "$errors verification check(s) failed" -Level ERROR
        return $false
    }
}

# Main installation flow
function Main {
    if ($Verify) {
        Test-Installation
        exit
    }
    
    New-VirtualEnv
    Install-PythonDeps
    Install-CliTools
    New-CacheDir
    Set-Environment
    $verified = Test-Installation
    
    if (!$Silent) {
        @"

╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ✓ Installation Complete!                                   ║
║                                                              ║
║   Cache directory: $CacheDir
║   Profile: $Profile
║   Architecture: $ARCH
║                                                              ║
║   Quick Start:                                               ║
║   PS> python -c "from ipfs_accelerate_py.common.llm_cache import get_global_llm_cache; cache = get_global_llm_cache(); print('Cache ready!')"
║                                                              ║
║   Documentation: CLI_INTEGRATIONS.md                         ║
║   Logs: $LogFile
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

"@
    }
    
    Write-ColorOutput -Message "Installation completed successfully" -Level SUCCESS
}

# Run main installation
Main
