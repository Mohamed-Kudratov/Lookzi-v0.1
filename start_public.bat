@echo off
title Lookzi — Public Server
cd /d "%~dp0"

echo.
echo  =====================================================
echo   Lookzi Virtual Try-On  ^|  Public Server (ngrok)
echo  =====================================================
echo.
echo  URL (doimiy, hech qachon o'zgarmaydi):
echo  https://gap-tiring-omit.ngrok-free.dev
echo.
echo  To'xtatish uchun: Ctrl+C
echo.

.venv\Scripts\python.exe app.py --preload --share --domain gap-tiring-omit.ngrok-free.dev

pause
