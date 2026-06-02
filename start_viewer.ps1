param(
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Ports = 8001..8010

function Test-ViewerApi {
    param([int]$Port)

    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/models" -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200
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
        if ($listener) {
            $listener.Stop()
        }
    }
}

function Wait-ViewerApi {
    param([int]$Port)

    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline) {
        if (Test-ViewerApi -Port $Port) {
            return $true
        }
        Start-Sleep -Milliseconds 300
    }
    return $false
}

$Port = $null
foreach ($candidate in $Ports) {
    if (Test-ViewerApi -Port $candidate) {
        $Port = $candidate
        break
    }
}

if (-not $Port) {
    $Python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $Python) {
        $Python = Get-Command py -ErrorAction SilentlyContinue
    }
    if (-not $Python) {
        throw "Python launcher was not found in PATH."
    }

    foreach ($candidate in $Ports) {
        if (-not (Test-PortFree -Port $candidate)) {
            continue
        }

        Start-Process `
            -WindowStyle Hidden `
            -FilePath $Python.Source `
            -ArgumentList @("viewer/server.py", "--host", "127.0.0.1", "--port", "$candidate") `
            -WorkingDirectory $Root

        if (Wait-ViewerApi -Port $candidate) {
            $Port = $candidate
            break
        }
    }
}

if (-not $Port) {
    throw "Could not start viewer API on ports 8001-8010."
}

$Url = "http://127.0.0.1:$Port/viewer/index.html"
Write-Host "Viewer server: $Url"

if (-not $NoOpen) {
    Start-Process $Url
}
