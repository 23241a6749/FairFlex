[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$webRoot = Join-Path $projectRoot "web"
$vite = Join-Path $webRoot "node_modules\.bin\vite.cmd"
$tsc = Join-Path $webRoot "node_modules\.bin\tsc.cmd"
$dataRoot = Join-Path $projectRoot "data\app"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment missing. Create it and install .[dev,app] first."
}
if (-not (Test-Path -LiteralPath $vite)) {
    throw "Web dependencies are missing. From the web folder, install the pinned dependencies before starting the app."
}
if (-not (Test-Path -LiteralPath $tsc)) {
    throw "TypeScript tooling is missing. From the web folder, install the pinned dependencies before starting the app."
}
$occupiedPort = Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -ne $occupiedPort) {
    throw "Port 8000 is already in use by process $($occupiedPort.OwningProcess). Stop the existing local server before launching FairFlex."
}
New-Item -ItemType Directory -Force -Path $dataRoot | Out-Null
$apiOut = Join-Path $dataRoot "api.out.log"
$apiErr = Join-Path $dataRoot "api.err.log"

Push-Location $webRoot
try {
    Write-Host "Building the verified local dashboard..."
    & $tsc -b
    if ($LASTEXITCODE -ne 0) { throw "TypeScript build failed." }
    & $vite build
    if ($LASTEXITCODE -ne 0) { throw "Web build failed." }
}
finally {
    Pop-Location
}

$api = Start-Process -FilePath $python -ArgumentList "-m", "fairflex.app.main" -WorkingDirectory $projectRoot -RedirectStandardOutput $apiOut -RedirectStandardError $apiErr -PassThru

try {
    $ready = $false
    foreach ($attempt in 1..20) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 2
            if ($health.status -eq "ready") { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) {
        throw "The API did not become ready. See $apiErr"
    }
    if ($api.HasExited) {
        throw "The API stopped unexpectedly. See $apiErr"
    }
    $dashboard = Invoke-WebRequest -Uri "http://127.0.0.1:8000/" -TimeoutSec 3
    if ($dashboard.StatusCode -ne 200) {
        throw "The built dashboard did not become available on the API origin."
    }

    Write-Host "FairFlex Demonstrator is running at http://127.0.0.1:8000"
    Write-Host "Offline controlled teaching simulation only - no physical charger control."
    Write-Host "Press Ctrl+C to stop the local demonstrator."
    Wait-Process -Id $api.Id
}
finally {
    if ($null -ne $api -and -not $api.HasExited) {
        Stop-Process -Id $api.Id -Force
    }
}
