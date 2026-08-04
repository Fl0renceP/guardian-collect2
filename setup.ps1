# One-shot local dev setup for Guardian Collective backend + frontend.
# Run from the repo root: .\setup.ps1
# Re-running is safe — it skips steps already done.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PYTHON = "py"
$PYTHON_VERSION = "-3.10-64"
# $PSScriptRoot can be empty if invoked via dot-source; fall back to MyInvocation
$REPO_ROOT = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Definition }
$VENV_DIR = "$REPO_ROOT\backend\.venv"

Write-Host "`n==> Checking Python 3.10..." -ForegroundColor Cyan
& $PYTHON $PYTHON_VERSION -c "import sys; print('  Python', sys.version)"

# --- Backend venv ---
if (-not (Test-Path "$VENV_DIR\Scripts\python.exe")) {
    Write-Host "`n==> Creating backend venv with Python 3.10..." -ForegroundColor Cyan
    & $PYTHON $PYTHON_VERSION -m venv $VENV_DIR
} else {
    Write-Host "`n==> Backend venv already exists, skipping creation." -ForegroundColor Green
}

$PIP = "$VENV_DIR\Scripts\pip.exe"

Write-Host "`n==> Upgrading pip..." -ForegroundColor Cyan
& $PIP install --upgrade pip --quiet

Write-Host "`n==> Installing backend requirements (this takes a while on first run)..." -ForegroundColor Cyan
& $PIP install -r "$REPO_ROOT\backend\requirements.txt"

Write-Host "`n==> Backend dependencies installed." -ForegroundColor Green

# --- Frontend ---
Write-Host "`n==> Installing frontend npm packages..." -ForegroundColor Cyan
npm --prefix "$REPO_ROOT\frontend" install

Write-Host "`n==> Frontend dependencies installed." -ForegroundColor Green

Write-Host @"

==========================================================
  Setup complete!

  To start the app:

  Backend:
    cd backend
    .\.venv\Scripts\Activate.ps1
    python app.py

  Frontend (separate terminal):
    cd frontend
    npm run dev

==========================================================
"@ -ForegroundColor Green
