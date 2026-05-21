# setup_server.ps1 - Run once to register Lookzi as an auto-start task
# Right-click -> "Run with PowerShell" (or run from admin PowerShell)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runnerBat = Join-Path $scriptDir "server_runner.bat"
$taskName  = "LookziAutoServer"

Write-Host ""
Write-Host "  Lookzi -- Full Server Setup" -ForegroundColor Cyan
Write-Host ""

# 1. Remove old startup shortcut
$startupLink = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Lookzi_Server.lnk"
if (Test-Path $startupLink) {
    Remove-Item $startupLink -Force
    Write-Host "  [OK] Removed old startup shortcut" -ForegroundColor Green
} else {
    Write-Host "  [--] No old startup shortcut found" -ForegroundColor DarkGray
}

# 2. Remove existing task
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -ne $existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "  [OK] Removed old task" -ForegroundColor Green
}

# 3. Build task
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$runnerBat`""

$trigger = New-ScheduledTaskTrigger -AtLogon -User $env:USERNAME
$trigger.Delay = "PT30S"

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit       ([TimeSpan]::Zero) `
    -RestartCount             10 `
    -RestartInterval          (New-TimeSpan -Minutes 1) `
    -MultipleInstances        IgnoreNew `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd

$principal = New-ScheduledTaskPrincipal `
    -UserId    $env:USERNAME `
    -LogonType Interactive `
    -RunLevel  Highest

# 4. Register
$result = Register-ScheduledTask `
    -TaskName   $taskName `
    -Action     $action `
    -Trigger    $trigger `
    -Settings   $settings `
    -Principal  $principal `
    -Description "Lookzi Virtual Try-On -- auto-start + crash recovery" `
    -Force

if ($null -ne $result) {
    Write-Host "  [OK] Task '$taskName' registered!" -ForegroundColor Green
} else {
    Write-Host "  [ERR] Registration failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  Setup complete!" -ForegroundColor Cyan
Write-Host ""
Write-Host "  The server will:" -ForegroundColor White
Write-Host "    - Start automatically when you log in (30s delay)" -ForegroundColor Green
Write-Host "    - Restart itself on crash (10 retries, 1 min apart)" -ForegroundColor Green
Write-Host "    - Log to: $scriptDir\logs\server.log" -ForegroundColor Green
Write-Host ""
Write-Host "  IMPORTANT: Enable Windows auto-login so PC logs in after power-on:" -ForegroundColor Yellow
Write-Host "    Win+R -> netplwiz -> uncheck password requirement" -ForegroundColor Yellow
Write-Host ""

# 5. Start now
Write-Host "  Starting server now..." -ForegroundColor Cyan
Start-ScheduledTask -TaskName $taskName
Write-Host "  [OK] Server started in background!" -ForegroundColor Green
Write-Host "  Logs: $scriptDir\logs\server.log" -ForegroundColor DarkGray
Write-Host "  Admin key: check server_config.json" -ForegroundColor DarkGray
Write-Host ""
