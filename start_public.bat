@echo off
title Lookzi — Public Server
cd /d "%~dp0"

echo.
echo  =====================================================
echo   Lookzi Virtual Try-On  ^|  Public Server (ngrok)
echo  =====================================================
echo.
echo  Ish PC ni server sifatida ishga tushirish uchun.
echo  Ngrok URL paydo bo'lgandan keyin uni uydan oching.
echo.
echo  To'xtatish uchun: Ctrl+C
echo.

.venv\Scripts\python.exe app.py --preload --share

pause
