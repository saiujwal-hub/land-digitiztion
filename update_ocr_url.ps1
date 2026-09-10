$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$LocalPy = "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe"

if (Test-Path $VenvPython) {
    & $VenvPython (Join-Path $ScriptDir "update_ocr_url.py") @args
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py (Join-Path $ScriptDir "update_ocr_url.py") @args
} elseif (Test-Path $LocalPy) {
    & $LocalPy (Join-Path $ScriptDir "update_ocr_url.py") @args
} else {
    & python (Join-Path $ScriptDir "update_ocr_url.py") @args
}
