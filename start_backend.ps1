# Quick Start Script for Data Analysis Tool (Windows PowerShell)

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host " Data Analysis Tool - Quick Start" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Prefer Python 3.12 (64-bit) when available to avoid scientific package issues on 3.13 x86.
$usePy312 = $false
try {
    py -3.12 --version *> $null
    if ($LASTEXITCODE -eq 0) {
        $usePy312 = $true
        $pythonVersion = py -3.12 --version 2>&1
        Write-Host "[✓] Python found (preferred): $pythonVersion" -ForegroundColor Green
    }
} catch {
}

if (-not $usePy312) {
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "[✓] Python found: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "[✗] ERROR: Python is not installed or not in PATH" -ForegroundColor Red
        Write-Host "Please install Python 3.12+ from https://www.python.org" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "[1/3] Setting up backend..." -ForegroundColor Yellow
Push-Location backend

# Load GROQ_API_KEY from project .env when not already present
if (-not $env:GROQ_API_KEY) {
    $envPath = Join-Path ".." ".env"
    if (Test-Path $envPath) {
        $line = Get-Content $envPath | Where-Object { $_ -match '^\s*GROQ_API_KEY\s*=' } | Select-Object -First 1
        if ($line) {
            $value = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'")
            if ($value) {
                $env:GROQ_API_KEY = $value
            }
        }
    }
}

if (-not $env:GROQ_API_KEY) {
    Write-Host "[!] WARNING: GROQ_API_KEY is not set. AI insights endpoint will fail until it is configured." -ForegroundColor Yellow
}

# Check if venv exists
if (!(Test-Path venv)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    if ($usePy312) {
        py -3.12 -m venv venv
    } else {
        python -m venv venv
    }
}

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "[✗] ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[2/3] Starting FastAPI server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "===============================================" -ForegroundColor Green
Write-Host "FastAPI server is starting..." -ForegroundColor Green
Write-Host "Server will be available at: http://localhost:8001" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: Keep this window open while using the app!" -ForegroundColor Yellow
Write-Host "===============================================" -ForegroundColor Green
Write-Host ""

# Start the server
if ($usePy312) {
    py -3.12 main.py
} else {
    python main.py
}
