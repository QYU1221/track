@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please create .venv with the runtime dependencies installed.
    popd
    exit /b 1
)
".venv\Scripts\python.exe" -m space_target_detection.main %*
set "runExit=%ERRORLEVEL%"
popd
exit /b %runExit%
