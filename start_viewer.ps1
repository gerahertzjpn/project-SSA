param(
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Ports = 8001..8010

function Test-PortOpen {
    param([int]$Port)
    try {
        $tcp = [System.Net.Sockets.TcpClient]::new()
        $ar = $tcp.BeginConnect([System.Net.IPAddress]::Loopback, $Port, $null, $null)
        $ok = $ar.AsyncWaitHandle.WaitOne(150)
        try { $tcp.EndConnect($ar) } catch {}
        $tcp.Dispose()
        return $ok
    } catch {
        return $false
    }
}

function Test-PortFree {
    param([int]$Port)
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse("127.0.0.1"), $Port)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) { $listener.Stop() }
    }
}

function Get-ViewerStatus {
    # Returns "fresh", "stale", "other", or "none"
    param([int]$Port)
    if (-not (Test-PortOpen -Port $Port)) { return "none" }
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/models" -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -ne 200) { return "other" }
        try { $body = $resp.Content | ConvertFrom-Json } catch { return "other" }
        if (-not ($body.PSObject.Properties.Name -contains "server")) { return "other" }
        if ($body.server.stale -eq $true) { return "stale" }
        return "fresh"
    } catch {
        return "other"
    }
}

function Wait-ViewerApi {
    param([int]$Port)
    $deadline = (Get-Date).AddSeconds(25)
    while ((Get-Date) -lt $deadline) {
        if ((Get-ViewerStatus -Port $Port) -eq "fresh") { return $true }
        Start-Sleep -Milliseconds 300
    }
    return $false
}

$Port = $null

# Phase 1: find a fresh server, or kill any stale viewer servers
foreach ($candidate in $Ports) {
    $status = Get-ViewerStatus -Port $candidate
    if ($status -eq "fresh") {
        $Port = $candidate
        break
    }
    if ($status -eq "stale") {
        Write-Host "Stale viewer on port $candidate (source changed) — restarting..."
        $conn = Get-NetTCPConnection -LocalPort $candidate -LocalAddress 127.0.0.1 -State Listen -ErrorAction SilentlyContinue
        if ($conn) {
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 300
        }
    }
}

# Phase 2: start a new server if no fresh server was found
if (-not $Port) {
    $Python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $Python) { $Python = Get-Command py -ErrorAction SilentlyContinue }
    if (-not $Python) { throw "Python was not found in PATH." }

    $LogDir = Join-Path $Root "viewer_logs"
    if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
    $LogOut = Join-Path $LogDir "server_out.log"
    $LogErr = Join-Path $LogDir "server_err.log"

    foreach ($candidate in $Ports) {
        if (-not (Test-PortFree -Port $candidate)) { continue }

        Write-Host "Starting viewer server on port $candidate..."

        $proc = Start-Process `
            -WindowStyle Hidden `
            -FilePath $Python.Source `
            -ArgumentList @("viewer/server.py", "--host", "127.0.0.1", "--port", "$candidate") `
            -WorkingDirectory $Root `
            -RedirectStandardOutput $LogOut `
            -RedirectStandardError $LogErr `
            -PassThru

        Start-Sleep -Milliseconds 500
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
