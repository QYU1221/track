$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw '请先创建包含本系统运行依赖的 .venv。'
}
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m space_target_detection.main @args
    $resultCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $resultCode
