<#
  Просмотр World-вьюера со ВТОРОГО ПК через Tailscale.

  Главный ПК держит весь бекенд (server.py, тренировки, LLM). Второй ПК открывает вьюер в своём
  браузере и рендерит pixi НАТИВНО (это не стрим экрана). Tailscale — приватная mesh-сеть: НЕ нужен
  проброс портов на роутере, только исходящее соединение с этого ПК.

  ОДИН РАЗ, вручную (это твой аккаунт — скрипт за тебя не залогинится):
    1. Установи Tailscale на ОБА ПК:  https://tailscale.com/download
    2. На обоих выполни:  tailscale up   (войти ОДНИМ аккаунтом)
    3. Для режима по умолчанию включи HTTPS в админке:
       https://login.tailscale.com/admin/dns  ->  HTTPS Certificates -> Enable

  Запуск (на ГЛАВНОМ ПК):
    .\start_viewer_remote.ps1                 # режим по умолчанию: tailscale serve (HTTPS, без фаервола)
    .\start_viewer_remote.ps1 -DirectIp       # альтернатива: сервер слушает Tailscale-IP (http://100.x.x.x)
    .\start_viewer_remote.ps1 -Stop           # выключить tailscale serve

  Полный разбор и траблшутинг:  docs/cloudcode/REMOTE_VIEWER.md
#>
param(
  [int]$Port = 8001,
  [switch]$DirectIp,
  [switch]$Stop
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Find-Tailscale {
  $cmd = Get-Command tailscale -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $exe = "C:\Program Files\Tailscale\tailscale.exe"
  if (Test-Path $exe) { return $exe }
  return $null
}

# Вызов tailscale: аргументы через массив, чтобы PowerShell не парсил '--флаги' как операторы.
function Invoke-TS {
  param([string]$Exe, [string[]]$TsArgs)
  & $Exe @TsArgs
}

$ts = Find-Tailscale
if (-not $ts) {
  Write-Host "Tailscale не найден." -ForegroundColor Yellow
  Write-Host "  1) Установи: https://tailscale.com/download"
  Write-Host "  2) Выполни 'tailscale up' на ОБОИХ ПК (один аккаунт)."
  Write-Host "  3) Запусти этот скрипт снова."
  exit 1
}

if ($Stop) {
  try { Invoke-TS $ts @("serve", "--https=443", "off") } catch {}
  try { Invoke-TS $ts @("serve", "reset") } catch {}
  Write-Host "tailscale serve выключен."
  exit 0
}

# проверяем, что 'tailscale up' выполнен — иначе IP не выдаётся
$tsIp = $null
try { $tsIp = (Invoke-TS $ts @("ip", "-4") | Select-Object -First 1) } catch {}
if ($tsIp) { $tsIp = "$tsIp".Trim() }
if (-not $tsIp) {
  Write-Host "Не получен Tailscale IP. Сначала выполни 'tailscale up' и войди в аккаунт." -ForegroundColor Yellow
  exit 1
}

if ($DirectIp) {
  # Сервер слушает ТОЛЬКО Tailscale-интерфейс (не виден в LAN). Может потребоваться правило брандмауэра.
  try {
    New-NetFirewallRule -DisplayName "SSA World Viewer (Tailscale) $Port" -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port -ErrorAction Stop | Out-Null
    Write-Host "Правило брандмауэра добавлено (входящие TCP $Port)."
  } catch {
    Write-Host "Не смог добавить правило брандмауэра (нужен запуск от админа). Если второй ПК не подключится —" -ForegroundColor Yellow
    Write-Host "открой PowerShell ОТ АДМИНА и выполни одну строку:"
    Write-Host ("  New-NetFirewallRule -DisplayName 'SSA World Viewer (Tailscale) {0}' -Direction Inbound -Action Allow -Protocol TCP -LocalPort {0}" -f $Port)
  }
  Write-Host ""
  Write-Host "URL для ВТОРОГО ПК (он тоже должен быть в твоём tailnet):" -ForegroundColor Green
  Write-Host ("  http://{0}:{1}/viewer_pixi/world.html" -f $tsIp, $Port)
  Write-Host ""
  Write-Host "Ctrl+C — остановить. Сервер работает в этом окне."
  & python (Join-Path $Root "viewer\server.py") "--host" $tsIp "--port" "$Port"
  exit $LASTEXITCODE
}

# --- режим по умолчанию: tailscale serve (проксирует tailnet -> localhost, HTTPS, без правил фаервола) ---
$dns = $null
try {
  $status = Invoke-TS $ts @("status", "--json") | ConvertFrom-Json
  if ($status -and $status.Self -and $status.Self.DNSName) { $dns = "$($status.Self.DNSName)".TrimEnd('.') }
} catch {}

Write-Host "Запускаю сервер на 127.0.0.1:$Port (в фоне)…"
$serverPy = Join-Path $Root "viewer\server.py"
$proc = Start-Process -FilePath "python" -ArgumentList @("`"$serverPy`"", "--host", "127.0.0.1", "--port", "$Port") -PassThru -WindowStyle Minimized
Start-Sleep -Seconds 2

Write-Host "Проксирую через 'tailscale serve' (HTTPS, только твой tailnet)…"
Invoke-TS $ts @("serve", "--bg", "$Port")
if ($LASTEXITCODE -ne 0) {
  Write-Host ""
  Write-Host "tailscale serve вернул ошибку. Частые причины:" -ForegroundColor Yellow
  Write-Host "  - HTTPS не включён: https://login.tailscale.com/admin/dns -> HTTPS Certificates -> Enable."
  Write-Host "  - Старая версия CLI: попробуй 'tailscale serve https / proxy $Port' или обнови Tailscale."
  Write-Host "  - Либо используй прямой режим:  .\start_viewer_remote.ps1 -DirectIp"
  try { Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue } catch {}
  exit 1
}

Write-Host ""
Write-Host "Готово. URL для ВТОРОГО ПК (он должен быть в твоём tailnet):" -ForegroundColor Green
if ($dns) {
  Write-Host ("  https://{0}/viewer_pixi/world.html" -f $dns)
} else {
  Write-Host "  https://<имя-этого-ПК>.<твой-tailnet>.ts.net/viewer_pixi/world.html"
  Write-Host "  (точное имя:  tailscale status)"
}
Write-Host ""
Write-Host ("Сервер работает в фоне (PID {0})." -f $proc.Id)
Write-Host "Выключить проксирование:  .\start_viewer_remote.ps1 -Stop"
Write-Host ("Остановить сервер:        Stop-Process -Id {0}" -f $proc.Id)
