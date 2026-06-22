param(
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Ports = 8001..8010
$StartTimeout = 30

function Test-PortOpen {
    param([int]$Port)
    try {
        $tcp = [System.Net.Sockets.TcpClient]::new()
        $ar = $tcp.BeginConnect([System.Net.IPAddress]::Loopback, $Port, $null, $null)
        $ok = $ar.AsyncWaitHandle.WaitOne(300)
        try { $tcp.EndConnect($ar) } catch {}
        $tcp.Dispose()
        return $ok
    } catch {
        return $false
    }
}

function Get-ViewerStatus {
    param([int]$Port)
    if (-not (Test-PortOpen -Port $Port)) { return "none" }
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/models" -UseBasicParsing -TimeoutSec 5
        if ($resp.StatusCode -ne 200) { return "other" }
        try { $body = $resp.Content | ConvertFrom-Json } catch { return "other" }
        if (-not ($body.PSObject.Properties.Name -contains "server")) { return "other" }
        if ($body.server.stale -eq $true) { return "stale" }
        return "fresh"
    } catch {
        return "other"
    }
}

function Invoke-KillPort {
    param([int]$Port)
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) { return }
    $ownerPids = @($conns | Select-Object -ExpandProperty OwningProcess -Unique)
    foreach ($procId in $ownerPids) {
        Write-Host "  Killing PID $procId on port $Port..."
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
        } catch {
            $null = & taskkill /F /PID $procId
        }
    }
    $deadline = (Get-Date).AddSeconds(6)
    while ((Get-Date) -lt $deadline) {
        $still = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if (-not $still) { return }
        Start-Sleep -Milliseconds 200
    }
    Write-Warning "Port $Port may still be in use after killing."
}

function Wait-ViewerApi {
    param([int]$Port)
    $deadline = (Get-Date).AddSeconds($StartTimeout)
    while ((Get-Date) -lt $deadline) {
        $status = Get-ViewerStatus -Port $Port
        if ($status -eq "fresh") { return $true }
        if ($status -eq "stale") { return $false }
        Start-Sleep -Milliseconds 400
    }
    return $false
}

# Phase 1: find a fresh server; kill every stale one
$FreshPort = $null

foreach ($candidate in $Ports) {
    $status = Get-ViewerStatus -Port $candidate
    if ($status -eq "fresh") {
        Write-Host "Found running fresh viewer on port $candidate."
        if (-not $FreshPort) { $FreshPort = $candidate }
    } elseif ($status -eq "stale") {
        Write-Host "Stale viewer on port $candidate (source changed) -- restarting..."
        Invoke-KillPort -Port $candidate
        if ($FreshPort -eq $candidate) { $FreshPort = $null }
    }
}

# Phase 2: start a new server if no fresh one survived
$Port = $FreshPort

if (-not $Port) {
    $Python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $Python) { $Python = Get-Command py -ErrorAction SilentlyContinue }
    if (-not $Python) { throw "Python was not found in PATH." }

    $LogDir = Join-Path $Root "viewer_logs"
    if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
    $LogOut = Join-Path $LogDir "server_out.log"
    $LogErr = Join-Path $LogDir "server_err.log"

    foreach ($candidate in $Ports) {
        $inUse = Get-NetTCPConnection -LocalPort $candidate -State Listen -ErrorAction SilentlyContinue
        if ($inUse) {
            Write-Host "Port $candidate still in use, skipping."
            continue
        }

        Write-Host "Starting viewer server on port $candidate..."

        $proc = Start-Process `
            -NoNewWindow `
            -FilePath $Python.Source `
            -ArgumentList @("viewer/server.py", "--host", "127.0.0.1", "--port", "$candidate") `
            -WorkingDirectory $Root `
            -RedirectStandardOutput $LogOut `
            -RedirectStandardError $LogErr `
            -PassThru

        Start-Sleep -Milliseconds 600
        if ($proc.HasExited) {
            $errText = if (Test-Path $LogErr) { (Get-Content $LogErr -Tail 5) -join "`n" } else { "(no log)" }
            Write-Warning "Server exited immediately (code $($proc.ExitCode)).`n$errText"
            continue
        }

        if (Wait-ViewerApi -Port $candidate) {
            $Port = $candidate
            break
        }

        Write-Warning "Server on port $candidate did not respond in time. See: $LogErr"
        try { $proc.Kill() } catch {}
    }
}

if (-not $Port) {
    throw "Could not start viewer. Check viewer_logs/server_err.log for details."
}

$Url = "http://127.0.0.1:$Port/viewer_pixi/home.html"
Write-Host "Viewer: $Url"

if (-not $NoOpen) {
    Start-Process $Url
}
