param(
    [int]$MaxTokens = 2048,
    [int]$UpdateFreq = 8,
    [int]$MaxEpoch = 30
)

$ErrorActionPreference = "Stop"

# Always run Docker Compose from the directory containing this script.
$projectRoot = $PSScriptRoot

# Fall back to Docker Desktop's per-user install path if PATH is stale.
$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerCommand) {
    $dockerExe = $dockerCommand.Source
}
else {
    $dockerExe = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"
}

if (-not (Test-Path -LiteralPath $dockerExe -PathType Leaf)) {
    throw "docker.exe was not found. Start Docker Desktop first."
}

Write-Host "HW5 RTX 3050 baseline" -ForegroundColor Cyan
Write-Host "max_tokens : $MaxTokens"
Write-Host "update_freq: $UpdateFreq"
Write-Host "effective tokens per update: $($MaxTokens * $UpdateFreq)"
Write-Host "max_epoch  : $MaxEpoch"
Write-Host "checkpoints: $projectRoot\checkpoints\rnn_baseline"

Push-Location $projectRoot
try {
    & $dockerExe compose run --rm hw5 fairseq-train /workspace/DATA/data-bin/ted2020 `
        --source-lang en `
        --target-lang zh `
        --arch lstm `
        --encoder-embed-dim 256 `
        --encoder-hidden-size 512 `
        --encoder-layers 1 `
        --decoder-embed-dim 256 `
        --decoder-hidden-size 512 `
        --decoder-out-embed-dim 256 `
        --decoder-layers 1 `
        --share-decoder-input-output-embed `
        --dropout 0.3 `
        --criterion label_smoothed_cross_entropy `
        --label-smoothing 0.1 `
        --optimizer adam `
        --adam-betas "(0.9,0.98)" `
        --adam-eps 1e-9 `
        --weight-decay 0.0001 `
        --lr 0.002 `
        --lr-scheduler inverse_sqrt `
        --warmup-updates 4000 `
        --warmup-init-lr 1e-7 `
        --clip-norm 1.0 `
        --max-tokens $MaxTokens `
        --update-freq $UpdateFreq `
        --max-epoch $MaxEpoch `
        --fp16 `
        --save-dir /workspace/checkpoints/rnn_baseline `
        --keep-last-epochs 5 `
        --log-format simple `
        --log-interval 50 `
        --num-workers 2

    if ($LASTEXITCODE -ne 0) {
        throw "Baseline training failed. Docker/fairseq exit code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
