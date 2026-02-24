# PowerShell script to run quantile policy tests
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvPath = Join-Path $ScriptPath "fica_env\Scripts"
$PythonPath = Join-Path $EnvPath "python.exe"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "QUANTILE POLICY - RUNNING TESTS" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Python Path: $PythonPath" -ForegroundColor Yellow
Write-Host ""

# Check if Python exists
if (-not (Test-Path $PythonPath)) {
    Write-Host "ERROR: Python not found at $PythonPath" -ForegroundColor Red
    Write-Host "Please ensure the FICA environment is properly installed." -ForegroundColor Red
    exit 1
}

# Run the tests
& $PythonPath "$ScriptPath\test_quantile.py"

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "ALL TESTS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Red
    Write-Host "SOME TESTS FAILED!" -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Red
}

exit $LASTEXITCODE
