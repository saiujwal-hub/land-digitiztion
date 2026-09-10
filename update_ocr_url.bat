@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    "%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%update_ocr_url.py" %*
    exit /b !ERRORLEVEL!
)

if exist "C:\Users\meesa\Downloads\final land\.venv\Scripts\python.exe" (
    "C:\Users\meesa\Downloads\final land\.venv\Scripts\python.exe" "%SCRIPT_DIR%update_ocr_url.py" %*
    exit /b !ERRORLEVEL!
)

py -0 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    py "%SCRIPT_DIR%update_ocr_url.py" %*
    exit /b !ERRORLEVEL!
)

if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
    "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" "%SCRIPT_DIR%update_ocr_url.py" %*
    exit /b !ERRORLEVEL!
)

python "%SCRIPT_DIR%update_ocr_url.py" %*
exit /b %ERRORLEVEL%
