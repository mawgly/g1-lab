$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.tools\python'
New-Item -ItemType Directory -Force '.tools' | Out-Null
if (-not (Test-Path '.tools\uv\uv.exe')) {
    Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '.tools\uv.zip'
    Expand-Archive -LiteralPath '.tools\uv.zip' -DestinationPath '.tools\uv' -Force
}
& '.tools\uv\uv.exe' python install 3.12
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed' }
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & '.tools\uv\uv.exe' venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
}
& '.tools\uv\uv.exe' pip sync --python '.venv\Scripts\python.exe' requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.venv\Scripts\python.exe' -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
Write-Host 'Ready. Open demo.cmd or manual.cmd.'
