@echo off
title Lookzi — Local Server
cd /d "%~dp0"

echo.
echo  =====================================================
echo   Lookzi Virtual Try-On  ^|  Local  (127.0.0.1:7860)
echo  =====================================================
echo.
echo  http://127.0.0.1:7860
echo.
echo  To'xtatish uchun: Ctrl+C
echo.

.venv\Scripts\python.exe app.py --preload

pause
