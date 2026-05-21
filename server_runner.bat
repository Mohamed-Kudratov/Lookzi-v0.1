@echo off
title Lookzi — Auto Server (crash-restart)
cd /d "%~dp0"
if not exist logs mkdir logs

echo. >> logs\server.log
echo ====================================================== >> logs\server.log
echo   Lookzi server_runner started: %date% %time%          >> logs\server.log
echo ====================================================== >> logs\server.log

:LOOP
echo. >> logs\server.log
echo [%date% %time%] ++ Starting Lookzi server... >> logs\server.log

.venv\Scripts\python.exe -u app.py --preload --share --domain gap-tiring-omit.ngrok-free.dev --log-file logs\server.log 2>> logs\server.log

set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] -- Server stopped (exit code: %EXIT_CODE%). Restarting in 15s... >> logs\server.log
timeout /t 15 /nobreak > nul
goto LOOP
