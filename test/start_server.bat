@echo off
title SAM 3 Segmentation Server
cd /d "%~dp0"

echo ============================================
echo   SAM 3 Interactive Segmentation Server
echo ============================================
echo.

REM --- Find conda Python ---
set "CONDA_PYTHON="

REM Try common conda installation paths
for %%p in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\miniconda" "%USERPROFILE%\Anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\miniconda") do (
    for %%e in (sam3 sam) do (
        if exist "%%~p\envs\%%e\python.exe" set "CONDA_PYTHON=%%~p\envs\%%e\python.exe"
    )
    if exist "%%~p\python.exe" set "CONDA_PYTHON=%%~p\python.exe"
)

if "%CONDA_PYTHON%"=="" (
    echo [ERROR] Cannot find conda environment.
    echo.
    echo Searched paths:
    for %%p in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\miniconda" "%USERPROFILE%\Anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\miniconda") do (
        echo   - %%p
    )
    echo.
    echo Please activate the environment manually, then run:
    echo   python main.py
    pause
    exit /b 1
)

echo [OK] Python: %CONDA_PYTHON%
echo.

REM --- Optional: Install dependencies (skip if already installed) ---
REM To enable, remove "REM" from the next 6 lines:
REM echo [1/3] Installing Python dependencies...
REM cd /d "%~dp0backend"
REM "%CONDA_PYTHON%" -m pip install -r requirements.txt -q
REM if %errorlevel% neq 0 (
REM     echo [WARN] pip install had issues, continuing anyway...
REM ) else (
REM     echo [OK] Dependencies ready.
REM )
REM echo.

REM --- Start backend ---
echo [1/2] Starting SAM 3 backend (loading model may take 30-60s)...
cd /d "%~dp0backend"
start "SAM3-Backend" cmd /c ""%CONDA_PYTHON%" main.py > backend.log 2>&1"
echo [OK] Backend starting on http://localhost:8501
echo.

REM --- Wait for server, then open browser ---
echo [2/2] Waiting for server... (8 seconds)
timeout /t 8 /nobreak >nul
start http://localhost:8501

echo.
echo ============================================
echo   Server is running!
echo   Open: http://localhost:8501
echo   Close this window to stop the server.
echo ============================================
echo.
echo Loading model... please wait 30-60s for first request.
pause
