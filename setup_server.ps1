# setup_server.ps1
# ─────────────────────────────────────────────────────────────────────────────
# Lookzi — One-time server setup
# Run this ONCE (as Administrator) to register the auto-start task.
# After this you never need to touch the bat files again.
# ─────────────────────────────────────────────────────────────────────────────

$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$runnerBat  = Join-Path $scriptDir "server_runner.bat"
$taskName   = "LookziAutoServer"

Write-Host ""
Write-Host "  ======================================================" -ForegroundColor Cyan
Write-Host "   Lookzi — Full Server Setup" -ForegroundColor Cyan
Write-Host "  ======================================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Remove old startup shortcut ───────────────────────────────────────
$startupLink = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Lookzi_Server.lnk"
if (Test-Path $startupLink) {
    Remove-Item $startupLink -Force
    Write-Host "  [OK] Removed old startup shortcut" -ForegroundColor Green
} else {
    Write-Host "  [--] No old startup shortcut (OK)" -ForegroundColor DarkGray
}

# ── 2. Remove existing scheduled task if present ─────────────────────────
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "  [OK] Removed existing task '$taskName'" -ForegroundColor Green
}

# ── 3. Build task components ──────────────────────────────────────────────
$action = New-ScheduledTaskAction `
    -Execute  "cmd.exe" `
    -Argument "/c `"$runnerBat`""

# At logon of current user, with a 30-second delay so desktop/drivers settle
$trigger = New-ScheduledTaskTrigger -AtLogon -User $env:USERNAME
$trigger.Delay = "PT30S"

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit       ([TimeSpan]::Zero)  `
    -RestartCount             10                  `
    -RestartInterval          (New-TimeSpan -Minutes 1) `
    -MultipleInstances        IgnoreNew           `
    -StartWhenAvailable                           `
    -DontStopIfGoingOnBatteries                   `
    -DontStopOnIdleEnd

$principal = New-ScheduledTaskPrincipal `
    -UserId   $env:USERNAME `
    -LogonType Interactive  `
    -RunLevel Highest

# ── 4. Register task ──────────────────────────────────────────────────────
Register-ScheduledTask `
    -TaskName   $taskName `
    -Action     $action `
    -Trigger    $trigger `
    -Settings   $settings `
    -Principal  $principal `
    -Description "Lookzi Virtual Try-On — auto-start + crash recovery" `
    -Force | Out-Null

Write-Host "  [OK] Task '$taskName' registered!" -ForegroundColor Green

# ── 5. Summary ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ======================================================" -ForegroundColor Cyan
Write-Host "   Setup complete!" -ForegroundColor Cyan
Write-Host "  ======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  The Lookzi server will now:" -ForegroundColor White
Write-Host "    * Start automatically when you log in" -ForegroundColor Green
Write-Host "    * Restart itself if it crashes (up to 10x, every 1 min)" -ForegroundColor Green
Write-Host "    * Log everything to: $scriptDir\logs\server.log" -ForegroundColor Green
Write-Host ""
Write-Host "  IMPORTANT — Enable Windows auto-login:" -ForegroundColor Yellow
Write-Host "    1. Press Win+R, type: netplwiz, press Enter" -ForegroundColor Yellow
Write-Host "    2. Uncheck 'Users must enter a password to use this computer'" -ForegroundColor Yellow
Write-Host "    3. Enter your password when prompted" -ForegroundColor Yellow
Write-Host "    4. Done — PC will log in automatically after power-on" -ForegroundColor Yellow
Write-Host ""

# ── 6. Start right now? ───────────────────────────────────────────────────
Write-Host "  Starting server NOW..." -ForegroundColor Cyan
Start-ScheduledTask -TaskName $taskName
Write-Host "  [OK] Server is starting in background!" -ForegroundColor Green
Write-Host ""
Write-Host "  Check logs: $scriptDir\logs\server.log" -ForegroundColor DarkGray
Write-Host "  Admin key:  see logs\server.log or server_config.json" -ForegroundColor DarkGray
Write-Host ""
