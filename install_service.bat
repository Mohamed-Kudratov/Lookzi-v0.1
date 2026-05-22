@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: install_service.bat  —  Lookzi ni Windows Service sifatida o'rnatish
:: Bir marta ishga tushiring. Admin huquqi so'raydi (UAC).
:: ─────────────────────────────────────────────────────────────────────────────
title Lookzi — Service Install

:: Self-elevate if not admin
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Admin huquqi so'ralyapti...
    powershell -Command "Start-Process cmd -ArgumentList '/c cd /d ""%~dp0"" && ""%~f0""' -Verb RunAs -Wait"
    exit /b
)

cd /d "%~dp0"

set NSSM=C:\Users\PC\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe
set PROJECT=C:\Users\PC\Desktop\Mohamed\FASHN VTON\fashn-vton-1.5
set PYTHON=%PROJECT%\.venv\Scripts\python.exe
set SERVICE=LookziServer
set DOMAIN=gap-tiring-omit.ngrok-free.dev
set TOKEN=3CvufXdO0qVqIrfMZf3WPTjCZyg_VNZf2E1DFwYTCTDwh796
set LOGFILE=logs\server.log

echo.
echo  =====================================================
echo   Lookzi -- Windows Service o'rnatish
echo  =====================================================
echo.

:: logs papkasini yaratish
if not exist "%PROJECT%\logs" mkdir "%PROJECT%\logs"
echo  [OK] logs\ papka tayyor

:: Eski startup shortcutni o'chirish
set LNK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Lookzi_Server.lnk
if exist "%LNK%" (
    del /f /q "%LNK%"
    echo  [OK] Startup shortcut o'chirildi
)

:: Agar service allaqachon bo'lsa — to'xtatib o'chirish
"%NSSM%" stop %SERVICE% >nul 2>&1
"%NSSM%" remove %SERVICE% confirm >nul 2>&1
echo  [OK] Eski service tozalandi

:: Service o'rnatish
echo  ... Service o'rnatilmoqda...
"%NSSM%" install %SERVICE% "%PYTHON%" "-u app.py --share --domain %DOMAIN% --authtoken %TOKEN% --log-file %LOGFILE%"
if %errorLevel% neq 0 (
    echo  [ERR] Service o'rnatilmadi!
    pause & exit /b 1
)

:: Service sozlamalari
"%NSSM%" set %SERVICE% AppDirectory "%PROJECT%"
"%NSSM%" set %SERVICE% DisplayName "Lookzi Virtual Try-On"
"%NSSM%" set %SERVICE% Description "Lookzi AI Virtual Try-On — auto server"
"%NSSM%" set %SERVICE% Start SERVICE_AUTO_START
"%NSSM%" set %SERVICE% ObjectName LocalSystem
"%NSSM%" set %SERVICE% AppStdout "%PROJECT%\logs\service_stdout.log"
"%NSSM%" set %SERVICE% AppStderr "%PROJECT%\logs\service_stderr.log"
"%NSSM%" set %SERVICE% AppRotateFiles 1
"%NSSM%" set %SERVICE% AppRotateBytes 5242880
"%NSSM%" set %SERVICE% AppRestartDelay 15000
echo  [OK] Sozlamalar saqlandi

:: Ishga tushirish
echo  ... Ishga tushirilmoqda...
"%NSSM%" start %SERVICE%

echo.
echo  =====================================================
echo   Tayyor!
echo  =====================================================
echo.
echo  Service nomi : LookziServer
echo  Holat        : Services.msc dan ko'rish mumkin
echo  Loglar       : %PROJECT%\logs\
echo.
echo  20 soniya kutib http://127.0.0.1:7860 ni oching
echo.
pause
