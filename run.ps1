# Single-command runner for Windows.
# Auto-installs Python + QQ NT via winget, fetches NapCatQQ, writes the OneBot
# WebSocket config matching .env, launches NapCat, and starts the bot.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$VenvDir       = ".venv"
$VenvPython    = Join-Path $PSScriptRoot "$VenvDir\Scripts\python.exe"
$NapCatDir     = Join-Path $PSScriptRoot "napcat"
$NapCatVersion = "v4.18.1"
$NapCatZipUrl  = "https://github.com/NapNeko/NapCatQQ/releases/download/$NapCatVersion/NapCat.Shell.zip"
$NapCatZipPath = Join-Path $env:TEMP "NapCat.Shell.zip"

$WingetPythonId = "Python.Python.3.12"
$WingetQQId     = "Tencent.QQ.NT"

function Refresh-Path {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "Machine")
}

function Ensure-Winget {
    if (Get-Command winget -ErrorAction SilentlyContinue) { return $true }
    Write-Warning "winget not available. Install 'App Installer' from the Microsoft Store, then rerun."
    return $false
}

function Ensure-Python {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { $cmd = Get-Command py -ErrorAction SilentlyContinue }
    if ($cmd) { return $cmd.Source }

    Write-Host "[run.ps1] Python not found - installing $WingetPythonId via winget"
    if (-not (Ensure-Winget)) {
        Write-Error "Install Python 3.10+ manually from https://www.python.org/, then rerun."
        exit 1
    }
    & winget install --id $WingetPythonId -e --source winget `
        --accept-source-agreements --accept-package-agreements --silent --scope user
    Refresh-Path
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Error "Python install completed but python.exe not on PATH. Open a new PowerShell window and rerun."
        exit 1
    }
    return $cmd.Source
}

function Ensure-Venv {
    if (-not (Test-Path $VenvPython)) {
        Write-Host "[run.ps1] creating virtualenv at $VenvDir"
        $pythonExe = Ensure-Python
        & $pythonExe -m venv $VenvDir
    }
    Write-Host "[run.ps1] installing Python requirements"
    & $VenvPython -m pip install --upgrade pip --quiet
    & $VenvPython -m pip install -r requirements.txt --quiet
}

function Ensure-DotEnv {
    if (-not (Test-Path ".env")) {
        Write-Host "[run.ps1] no .env found - copying from .env.example."
        Copy-Item ".env.example" ".env"
        Write-Host "Edit .env (DEEPSEEK_API_KEY, ALLOWED_GROUPS at minimum), then rerun." -ForegroundColor Yellow
        exit 1
    }
}

function Ensure-NapCat {
    $launcher = Join-Path $NapCatDir "launcher.bat"
    if (Test-Path $launcher) {
        Write-Host "[run.ps1] NapCatQQ already installed at $NapCatDir"
        return
    }
    Write-Host "[run.ps1] downloading NapCatQQ $NapCatVersion from $NapCatZipUrl"
    $oldPref = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"
    try {
        Invoke-WebRequest -Uri $NapCatZipUrl -OutFile $NapCatZipPath -UseBasicParsing
    } finally {
        $ProgressPreference = $oldPref
    }
    Write-Host "[run.ps1] extracting to $NapCatDir"
    if (-not (Test-Path $NapCatDir)) {
        New-Item -ItemType Directory -Path $NapCatDir | Out-Null
    }
    Expand-Archive -Path $NapCatZipPath -DestinationPath $NapCatDir -Force
    Remove-Item $NapCatZipPath -ErrorAction SilentlyContinue
}

function Find-QQExe {
    $candidates = @(
        "${env:ProgramFiles}\Tencent\QQNT\QQ.exe",
        "${env:ProgramFiles(x86)}\Tencent\QQNT\QQ.exe",
        "${env:LOCALAPPDATA}\Tencent\QQNT\QQ.exe"
    )
    foreach ($p in $candidates) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    return $null
}

function Ensure-QQ {
    if (Find-QQExe) { return $true }
    Write-Host "[run.ps1] QQ NT not found - installing $WingetQQId via winget (may prompt for admin)"
    if (-not (Ensure-Winget)) {
        Write-Warning "Install QQ manually from https://im.qq.com/ then rerun."
        return $false
    }
    & winget install --id $WingetQQId -e --source winget `
        --accept-source-agreements --accept-package-agreements --silent
    if (-not (Find-QQExe)) {
        Write-Warning "winget reported success but QQ.exe still not found. Install manually and rerun."
        return $false
    }
    return $true
}

function Read-EnvValue([string]$key) {
    if (-not (Test-Path ".env")) { return "" }
    $line = Select-String -Path ".env" -Pattern "^\s*$key\s*=\s*(.*)$" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $line) { return "" }
    return $line.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
}

function Test-NapCatRunning {
    return [bool](Get-Process -Name "QQ" -ErrorAction SilentlyContinue)
}

function Test-NapCatConfigured {
    $cfgDir = Join-Path $NapCatDir "config"
    if (-not (Test-Path $cfgDir)) { return $false }
    $files = Get-ChildItem -Path $cfgDir -Filter "onebot11_*.json" -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        try { $cfg = Get-Content $f.FullName -Raw | ConvertFrom-Json } catch { continue }
        if ($null -eq $cfg.network) { continue }
        foreach ($key in @("websocketServers", "websocketClients", "httpServers", "httpSseServers")) {
            $arr = $cfg.network.$key
            if ($null -ne $arr) {
                foreach ($entry in @($arr)) {
                    if ($entry.enable) { return $true }
                }
            }
        }
    }
    return $false
}

# Write a WebSocket entry into every napcat\config\onebot11_*.json that doesn't
# already have one enabled. Forward mode -> websocketServers; reverse -> websocketClients.
# Returns the list of files modified.
function Write-NapCatWsConfig {
    $cfgDir = Join-Path $NapCatDir "config"
    if (-not (Test-Path $cfgDir)) { return @() }
    $files = Get-ChildItem -Path $cfgDir -Filter "onebot11_*.json" -ErrorAction SilentlyContinue
    if (-not $files) { return @() }

    $mode = (Read-EnvValue "ONEBOT_MODE")
    if (-not $mode) { $mode = "forward" }
    $mode = $mode.ToLower()
    $accessToken = Read-EnvValue "ONEBOT_ACCESS_TOKEN"

    $modified = @()
    foreach ($f in $files) {
        try { $cfg = Get-Content $f.FullName -Raw | ConvertFrom-Json } catch {
            Write-Warning "couldn't parse $($f.Name) - skipping"
            continue
        }
        if ($null -eq $cfg.network) { continue }

        if ($mode -eq "forward") {
            $wsUrl = Read-EnvValue "ONEBOT_WS_URL"
            if (-not $wsUrl) { $wsUrl = "ws://127.0.0.1:3001" }
            try {
                $uri = [System.Uri]::new($wsUrl)
                $wsHost = $uri.Host
                $wsPort = $uri.Port
            } catch { $wsHost = "127.0.0.1"; $wsPort = 3001 }
            if (-not $wsHost) { $wsHost = "127.0.0.1" }

            $entry = [PSCustomObject]@{
                enable               = $true
                name                 = "bot-auto"
                host                 = $wsHost
                port                 = [int]$wsPort
                reportSelfMessage    = $false
                enableForcePushEvent = $true
                messagePostFormat    = "array"
                token                = $accessToken
                debug                = $false
                heartInterval        = 30000
            }
            $existing = @($cfg.network.websocketServers)
            $found = $false
            for ($i = 0; $i -lt $existing.Count; $i++) {
                if ($existing[$i] -and $existing[$i].name -eq "bot-auto") {
                    $existing[$i] = $entry; $found = $true; break
                }
            }
            if (-not $found) { $existing = @($existing | Where-Object { $_ }) + $entry }
            $cfg.network.websocketServers = $existing
        }
        else {
            $revHost = Read-EnvValue "ONEBOT_REVERSE_HOST"
            $revPort = Read-EnvValue "ONEBOT_REVERSE_PORT"
            $revPath = Read-EnvValue "ONEBOT_REVERSE_PATH"
            if (-not $revHost -or $revHost -eq "0.0.0.0") { $revHost = "127.0.0.1" }
            if (-not $revPort) { $revPort = "3001" }
            if (-not $revPath) { $revPath = "/onebot/v11/ws" }
            $url = "ws://${revHost}:${revPort}${revPath}"

            $entry = [PSCustomObject]@{
                enable            = $true
                name              = "bot-auto"
                url               = $url
                reportSelfMessage = $false
                messagePostFormat = "array"
                token             = $accessToken
                debug             = $false
                heartInterval     = 30000
                reconnectInterval = 5000
            }
            $existing = @($cfg.network.websocketClients)
            $found = $false
            for ($i = 0; $i -lt $existing.Count; $i++) {
                if ($existing[$i] -and $existing[$i].name -eq "bot-auto") {
                    $existing[$i] = $entry; $found = $true; break
                }
            }
            if (-not $found) { $existing = @($existing | Where-Object { $_ }) + $entry }
            $cfg.network.websocketClients = $existing
        }

        $json = $cfg | ConvertTo-Json -Depth 32
        [System.IO.File]::WriteAllText($f.FullName, $json, [System.Text.UTF8Encoding]::new($false))
        Write-Host "[run.ps1] wrote $mode WebSocket config into $($f.Name)"
        $modified += $f.FullName
    }
    return $modified
}

function Start-NapCat {
    if (Test-NapCatRunning) {
        Write-Host "[run.ps1] QQ.exe already running - assuming NapCat is up, skipping launch"
        return
    }
    if (-not (Find-QQExe)) {
        Write-Warning "QQ NT not found - skipping NapCat launch."
        return
    }
    $build = [System.Environment]::OSVersion.Version.Build
    $launcherName = if ($build -ge 22000) { "launcher.bat" } else { "launcher-win10.bat" }
    $launcher = Join-Path $NapCatDir $launcherName
    if (-not (Test-Path $launcher)) {
        Write-Warning "$launcherName not found in $NapCatDir - skipping NapCat launch."
        return
    }
    $uin = Read-EnvValue "NAPCAT_QQ_UIN"
    Write-Host "[run.ps1] launching $launcherName$(if ($uin) { " $uin" }) in a new window"
    if ($uin) {
        Start-Process -FilePath $launcher -ArgumentList $uin -WorkingDirectory $NapCatDir
    } else {
        Start-Process -FilePath $launcher -WorkingDirectory $NapCatDir
    }
}

# ---- main ----
Ensure-Venv
Ensure-DotEnv
Ensure-NapCat
Ensure-QQ | Out-Null

$napcatConfigured = Test-NapCatConfigured

if (-not $napcatConfigured) {
    # See if NapCat has been logged into at least once (config file exists).
    $cfgDir = Join-Path $NapCatDir "config"
    $onebotFiles = @(Get-ChildItem -Path $cfgDir -Filter "onebot11_*.json" -ErrorAction SilentlyContinue)

    if ($onebotFiles.Count -gt 0) {
        # Logged in but no WS - write it now.
        $wrote = Write-NapCatWsConfig
        if ($wrote.Count -gt 0) {
            Write-Host ""
            if (Test-NapCatRunning) {
                Write-Host "===== Reload NapCat network =====" -ForegroundColor Yellow
                Write-Host "WebSocket config written, but NapCat is still using the old in-memory config."
                Write-Host "Either:"
                Write-Host "  (a) Open NapCat WebUI -> Network Config -> click reload/restart, OR"
                Write-Host "  (b) Close the NapCat window and rerun setup.bat (quick-login)."
                Write-Host "================================" -ForegroundColor Yellow
                exit 0
            } else {
                Start-NapCat
                Write-Host ""
                Write-Host "NapCat launched with the new WS config. Wait for QQ login to finish, then rerun setup.bat to start the bot." -ForegroundColor Yellow
                exit 0
            }
        }
    }

    # First-time: no onebot config yet. Launch NapCat for QR scan.
    Start-NapCat
    Write-Host ""
    Write-Host "===== NapCat first-run setup =====" -ForegroundColor Yellow
    Write-Host "1. In the new NapCat window, scan the QR code with the QQ mobile app."
    Write-Host "2. After login, rerun setup.bat - it will auto-write the WS config based on .env."
    Write-Host "3. Tip: add NAPCAT_QQ_UIN=<your-qq-number> to .env to enable quick-login next time."
    Write-Host "===================================" -ForegroundColor Yellow
    exit 0
}

# Configured. Make sure NapCat is up, then start the bot.
if (-not (Test-NapCatRunning)) { Start-NapCat }

Write-Host "[run.ps1] NapCat already configured - starting bot"
# Python's logger writes to stderr by default. Under $ErrorActionPreference=Stop
# every stderr line is wrapped as a NativeCommandError and kills the script.
$ErrorActionPreference = "Continue"
& $VenvPython main.py
exit $LASTEXITCODE
