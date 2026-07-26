@echo off
cls
cd /d "C:\Users\%USERNAME%\odysseus"

:menu
echo ===================================================
echo               ODYSSEUS MANAGEMENT MENU
echo ===================================================
echo  1. Update Odysseus and Start Normally
echo  2. Enter Virtual Environment (venv) for pip updates
echo  3. Update All Ollama Models
echo  4. Exit
echo ===================================================
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto option1
if "%choice%"=="2" goto option2
if "%choice%"=="3" goto option3
if "%choice%"=="4" exit
echo Invalid choice, please try again. & timeout /t 2 >nul & cls & goto menu

:option1
echo Running git pull...
git pull
echo.
echo Odysseus update ready. Press Ctrl+C now to abort, or any key to start...
pause
goto start_odysseus

:option2
echo Opening command prompt with venv activated...
:: Spawns a new CMD window, activates the venv, and keeps it open for you
start "" cmd /k "cd /d C:\Users\%USERNAME%\odysseus && .\venv\Scripts\activate"
cls
goto menu

:option3
echo ===================================================
echo Updating all Ollama models...
echo ===================================================
powershell -ExecutionPolicy Bypass -Command "ollama list | Select-Object -Skip 1 | ConvertFrom-String | %% { write-host 'Updating' $_.p1 '...'; ollama pull $_.p1 }"
echo.
echo Model updates completed!
echo.
pause
cls
goto menu

:start_odysseus
start "" powershell.exe -ExecutionPolicy Bypass -File ".\launch-windows.ps1"

echo Waiting for http://127.0.0.1:7000 ...
:waitloop
powershell -Command "try { (Invoke-WebRequest -Uri http://127.0.0.1:7000 -UseBasicParsing) > $null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    timeout /t 2 >nul
    goto waitloop
)

echo Launching Firefox...
start "" "C:\Program Files\Mozilla Firefox\firefox.exe" http://127.0.0.1:7000
exit