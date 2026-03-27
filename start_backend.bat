@echo off
REM Quick Start Script for Data Analysis Tool (Windows)

set "PY_CMD=python"
py -3.12 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py -3.12"
)

echo.
echo ===============================================
echo  Data Analysis Tool - Quick Start
echo ===============================================
echo.

REM Check if Python is installed
%PY_CMD% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.12+ from https://www.python.org
    pause
    exit /b 1
)

echo [1/3] Setting up backend...
cd backend

REM Load GROQ_API_KEY from project .env when not already set
if not defined GROQ_API_KEY (
    if exist ..\.env (
        for /f "tokens=1,* delims==" %%A in ('findstr /B /C:"GROQ_API_KEY=" ..\.env') do (
            set "GROQ_API_KEY=%%B"
        )
    )
)

if not defined GROQ_API_KEY (
    echo WARNING: GROQ_API_KEY is not set. AI insights endpoint will fail until it is configured.
)

REM Check if venv exists, if not create it
if not exist venv (
    echo Creating virtual environment...
    %PY_CMD% -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt >nul 2>&1

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/3] Starting FastAPI server...
echo.
echo ===============================================
echo FastAPI server is starting...
echo Server will be available at: http://localhost:8001
echo.
echo IMPORTANT: Keep this window open while using the app!
echo ===============================================
echo.

REM Start the server
%PY_CMD% main.py
