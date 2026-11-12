param(
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'

if (-not $SkipBuild) {
    $buildScript = Join-Path $PSScriptRoot 'cargo-local.ps1'
    & $buildScript build --release --offline --bin space_target_detection

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$packageDir = Join-Path $PSScriptRoot 'dist'
New-Item -ItemType Directory -Path $packageDir -Force | Out-Null

$compiledProgram = Join-Path $PSScriptRoot 'target/release/space_target_detection.exe'
$packagedProgram = Join-Path $packageDir 'space_target_detection.exe'
Copy-Item -LiteralPath $compiledProgram -Destination $packagedProgram -Force

$resourceRoot = Join-Path $PSScriptRoot 'resources'
foreach ($file in Get-ChildItem -LiteralPath $resourceRoot -Recurse -File) {
    $relative = $file.FullName.Substring($resourceRoot.Length + 1)
    $destination = Join-Path (Join-Path $packageDir 'resources') $relative

    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($destination)) -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
}

foreach ($fileName in @('README.md', 'THIRD_PARTY_NOTICES_NUMPY.txt')) {
    $sourcePath = Join-Path $PSScriptRoot $fileName
    $destinationPath = Join-Path $packageDir $fileName
    Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Force
}

Write-Output "Native package: $packageDir"
