param(
    [int]$MaxEpoch = 30,
    [int]$MaxTokens = 2048,
    [int]$AccumSteps = 8,
    [switch]$SmokeTest,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

# 這個 .ps1 只是 Windows → Docker → Python 的快捷入口。
# 真正的模型、Attention 與訓練迴圈都在 hw05_teacher_baseline.py。
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
    "hw05_teacher_baseline.py",
    "--max-epoch", $MaxEpoch,
    "--max-tokens", $MaxTokens,
    "--accum-steps", $AccumSteps
)
if ($SmokeTest) {
    $pythonArguments += "--smoke-test"
}
if ($NoResume) {
    $pythonArguments += "--no-resume"
}

Write-Host "HW5 teacher-style GRU + Attention baseline" -ForegroundColor Cyan
Write-Host "Python source : $projectRoot\hw05_teacher_baseline.py"
Write-Host "Max epoch     : $MaxEpoch"
Write-Host "Max tokens    : $MaxTokens"
Write-Host "Accum steps   : $AccumSteps"
Write-Host "Checkpoint    : $projectRoot\checkpoints\teacher_gru_baseline"

Push-Location $projectRoot
try {
    & $dockerExe compose run --rm hw5 python @pythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Teacher baseline failed. Docker/Python exit code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
