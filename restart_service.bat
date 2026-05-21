@echo off
net session >nul 2>&1
if %errorLevel% neq 0 (
    powershell -Command "Start-Process cmd -ArgumentList '/c cd /d ""%~dp0"" && ""%~f0""' -Verb RunAs -Wait"
    exit /b
)
net stop LookziServer
timeout /t 3 /nobreak >nul
net start LookziServer
echo Lookzi service qayta ishga tushdi!
timeout /t 3 /nobreak >nul
