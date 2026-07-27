@echo off
title SD3.5 Server

echo =====================================
echo Activating AI virtual environment...
echo =====================================

call C:\Users\%username%\odysseus\ai_venv\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Python:
python --version

echo.
echo Starting SD3.5 server...
echo =====================================

cd /d C:\Users\%username%\odysseus\sd_server_dir

REM python sd_server.py
python unified_ai_studio.py

echo.
echo Server stopped.
pause