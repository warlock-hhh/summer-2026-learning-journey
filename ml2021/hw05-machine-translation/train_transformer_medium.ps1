param(
    [int]$MaxEpoch = 30,
    [int]$MaxTokens = 1536,
    [int]$AccumSteps = 8,
    [switch]$SmokeTest,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerCommand) {
    $dockerExe = $dockerCommand.Source
}
else {
    $dockerExe = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"
}

if (-not (Test-Path -LiteralPath $dockerExe -PathType Leaf)) {
    throw "docker.exe was not found. Install or start Docker Desktop first."
}

$pythonArguments = @(
    "hw05_transformer_medium.py",
    "--max-epoch", $MaxEpoch,
    "--max-tokens", $MaxTokens,
    "--accum-steps", $AccumSteps
)
if ($SmokeTest) { $pythonArguments += "--smoke-test" }
if ($NoResume) { $pythonArguments += "--no-resume" }

Write-Host "HW5 Transformer Medium Baseline" -ForegroundColor Cyan
Write-Host "Architecture : 4-layer Encoder + 4-layer Decoder, d_model=256, FFN=1024, heads=4"
Write-Host "Max epoch    : $MaxEpoch"
Write-Host "Max tokens   : $MaxTokens"
Write-Host "Accum steps  : $AccumSteps"
Write-Host "Checkpoint   : $projectRoot\checkpoints\transformer_medium"

Push-Location $projectRoot
try {
    & $dockerExe compose run --rm hw5 python @pythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Transformer training failed. Docker/Python exit code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
