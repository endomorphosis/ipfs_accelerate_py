# Uninstaller for ipfs_accelerate_py Cache Infrastructure (Windows)

param(
    [switch]$Silent = $false
)

$ErrorActionPreference = "Stop"

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    if (!$Silent) {
        switch ($Level) {
            "INFO"    { Write-Host "→ $Message" -ForegroundColor Cyan }
            "SUCCESS" { Write-Host "✓ $Message" -ForegroundColor Green }
            "WARN"    { Write-Host "⚠ $Message" -ForegroundColor Yellow }
            "ERROR"   { Write-Host "✗ $Message" -ForegroundColor Red }
        }
    }
}

# Banner
if (!$Silent) {
    @"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ipfs_accelerate_py Cache Infrastructure Uninstaller        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

"@
}

# Ask for confirmation
if (!$Silent) {
    $response = Read-Host "Are you sure you want to uninstall ipfs_accelerate_py cache infrastructure? (y/N)"
    if ($response -ne "y") {
        Write-ColorOutput -Message "Uninstall cancelled" -Level SUCCESS
        exit 0
    }
}

Write-ColorOutput -Message "Uninstalling..." -Level WARN

# Find Python
$pythonCmd = $null
foreach ($py in @("python3", "python", "py")) {
    try {
        $null = & $py --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $py
            break
        }
    } catch {
        continue
    }
}

# Uninstall Python package
if ($null -ne $pythonCmd) {
    Write-ColorOutput -Message "Uninstalling Python package..." -Level INFO
    try {
        & $pythonCmd -m pip uninstall -y ipfs_accelerate_py 2>&1 | Out-Null
    } catch {
        Write-ColorOutput -Message "Could not uninstall package (may not be installed)" -Level WARN
    }
}

# Remove cache directory
$cacheDir = "$env:USERPROFILE\.cache\ipfs_accelerate_py"
if (Test-Path $cacheDir) {
    if ($Silent) {
        Remove-Item -Recurse -Force $cacheDir
    } else {
        $response = Read-Host "Remove cache directory ($cacheDir)? (y/N)"
        if ($response -eq "y") {
            Write-ColorOutput -Message "Removing cache directory..." -Level INFO
            Remove-Item -Recurse -Force $cacheDir
            Write-ColorOutput -Message "Cache directory removed" -Level SUCCESS
        }
    }
}

# Remove environment variables
Write-ColorOutput -Message "Removing environment variables..." -Level INFO
[Environment]::SetEnvironmentVariable("IPFS_ACCELERATE_CACHE_DIR", $null, "User")
[Environment]::SetEnvironmentVariable("IPFS_ACCELERATE_CACHE_TTL", $null, "User")
Write-ColorOutput -Message "Environment variables removed" -Level SUCCESS

# Remove virtual environment
if (Test-Path ".venv") {
    if ($Silent) {
        Remove-Item -Recurse -Force ".venv"
    } else {
        $response = Read-Host "Remove virtual environment (.venv)? (y/N)"
        if ($response -eq "y") {
            Write-ColorOutput -Message "Removing virtual environment..." -Level INFO
            Remove-Item -Recurse -Force ".venv"
            Write-ColorOutput -Message "Virtual environment removed" -Level SUCCESS
        }
    }
}

if (!$Silent) {
    @"

╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ✓ Uninstallation Complete!                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

"@
}
