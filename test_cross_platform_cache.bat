@echo off
REM Cross-Platform Cache Test - Windows Batch Script
REM This script helps Windows users run the cross-platform cache test

echo ============================================
echo Cross-Platform Cache Test (Windows)
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [INFO] Virtual environment not found, creating one...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
    echo.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [OK] Virtual environment activated
echo.

REM Check if dependencies are installed
echo [INFO] Checking dependencies...
python -c "import cryptography" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Dependencies not installed, installing now...
    echo.
    python -m pip install --upgrade pip
    pip install cryptography py-multiformats-cid
    
    REM Try to install libp2p (may fail on Windows)
    echo [INFO] Attempting to install libp2p (may not work on Windows)...
    pip install libp2p>=0.4.0 pymultihash>=0.8.2 2>nul
    if errorlevel 1 (
        echo [WARN] libp2p installation failed (expected on Windows)
        echo [INFO] Will continue without P2P support
    )
    echo.
)

echo [OK] Dependencies checked
echo.

REM Run the cross-platform test
echo [INFO] Running cross-platform test...
echo ============================================
echo.

python test_cross_platform_cache.py

if errorlevel 1 (
    echo.
    echo ============================================
    echo [ERROR] Test failed with errors
    echo.
    echo Review the output above for details.
    echo Common Windows issues:
    echo   1. libp2p may not be available (this is OK)
    echo   2. Some P2P tests may be skipped
    echo   3. Ensure Python is from python.org, not Microsoft Store
    echo.
    echo See CROSS_PLATFORM_TESTING_GUIDE.md for help
    echo ============================================
) else (
    echo.
    echo ============================================
    echo [SUCCESS] Test completed
    echo.
    echo Review the Platform Compatibility Report above
    echo.
    echo Next steps:
    echo   1. If tests passed, you can proceed to Docker testing
    echo   2. If P2P tests failed, you can still use cache without P2P
    echo   3. See CROSS_PLATFORM_TESTING_GUIDE.md for more info
    echo ============================================
)

echo.
pause
