@echo off
REM Launches the AAL Wheel Advisor dashboard (starts the Flask server if it
REM isn't already running, then opens it in the default browser).
setlocal
set "PROJECT_DIR=%~dp0.."
cd /d "%PROJECT_DIR%"

netstat -ano | findstr ":5000" | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo Dashboard server already running on port 5000.
) else (
    echo Starting AAL Wheel Advisor dashboard server...
    start "AAL Wheel Dashboard Server" /min cmd /c "python dashboard\app.py"
    timeout /t 3 /nobreak >nul
)

start "" http://127.0.0.1:5000
