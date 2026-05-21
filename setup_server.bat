@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: setup_server.bat  —  Run ONCE to install Lookzi as a Windows auto-start task
:: Double-click this file (it will ask for Admin access once via UAC)
:: ─────────────────────────────────────────────────────────────────────────────
title Lookzi Server Setup

:: Self-elevate if not already running as Administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrator access...
    powershell -Command "Start-Process cmd -ArgumentList '/c cd /d ""%~dp0"" && ""%~f0""' -Verb RunAs -Wait"
    exit /b
)

cd /d "%~dp0"

echo.
echo  ======================================================
echo   Lookzi -- Full Server Setup
echo  ======================================================
echo.

:: ── Remove old startup shortcut ────────────────────────────────────────────
set LNK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Lookzi_Server.lnk
if exist "%LNK%" (
    del /f /q "%LNK%"
    echo  [OK] Removed old startup shortcut
) else (
    echo  [--] No old startup shortcut found
)

:: ── Remove existing scheduled task ─────────────────────────────────────────
schtasks /delete /tn "LookziAutoServer" /f >nul 2>&1
echo  [OK] Cleared any existing task

:: ── Register new task ───────────────────────────────────────────────────────
:: Runs server_runner.bat at logon with 30s delay
schtasks /create ^
  /tn "LookziAutoServer" ^
  /tr "cmd.exe /c \"\"%~dp0server_runner.bat\"\"" ^
  /sc ONLOGON ^
  /delay 0000:30 ^
  /rl HIGHEST ^
  /f

if %errorLevel% neq 0 (
    echo.
    echo  [ERR] Task registration failed!
    pause
    exit /b 1
)

echo.
echo  [OK] Task "LookziAutoServer" registered!
echo.
echo  ======================================================
echo   Setup complete!
echo  ======================================================
echo.
echo  The server will now:
echo    - Start automatically when you log in (30s delay)
echo    - Restart itself if it crashes
echo    - Log to:  %~dp0logs\server.log
echo.
echo  IMPORTANT - Enable auto-login so PC logs in after power-on:
echo    Win+R  -^>  netplwiz  -^>  uncheck password requirement
echo.
echo  Starting server now...
schtasks /run /tn "LookziAutoServer"
echo  [OK] Server started in background!
echo.
echo  Admin key: check %~dp0server_config.json
echo  Logs:      %~dp0logs\server.log
echo.
pause
