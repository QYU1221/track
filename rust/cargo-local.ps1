param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CargoArgs
)

$ErrorActionPreference = 'Stop'

$env:RUSTUP_HOME = Join-Path $PSScriptRoot '.toolchain\rustup'
$env:CARGO_HOME = Join-Path $PSScriptRoot '.toolchain\cargo'
$env:PATH = "$env:CARGO_HOME\bin;$env:PATH"

Push-Location $PSScriptRoot
try {
    $cargoExe = Join-Path $env:CARGO_HOME 'bin\cargo.exe'
    & $cargoExe @CargoArgs
    $result = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $result
