param(
    [switch]$StagedOnly,
    [switch]$History
)

$ErrorActionPreference = "Stop"

# Common provider tokens, private-key headers, and direct secret assignments.
$patterns = @(
    'ghp_[A-Za-z0-9]{20,}',
    'github_pat_[A-Za-z0-9_]{20,}',
    'sk-[A-Za-z0-9_-]{20,}',
    'AKIA[0-9A-Z]{16}',
    'AIza[0-9A-Za-z_-]{30,}',
    '-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----',
    '(api[_-]?key|access[_-]?token|client[_-]?secret|password)[[:space:]]*[:=][[:space:]]*[^[:space:]]{12,}'
)
$pattern = '(' + ($patterns -join '|') + ')'

function Test-Revision {
    param(
        [string]$Revision,
        [switch]$Cached
    )

    if ($Cached) {
        $result = & git grep --cached -I -n -E $pattern -- 2>$null
    }
    else {
        $result = & git grep -I -n -E $pattern $Revision -- 2>$null
    }

    if ($LASTEXITCODE -eq 0) {
        return @($result)
    }
    if ($LASTEXITCODE -eq 1) {
        return @()
    }
    throw "git grep failed for: $Revision"
}

& git rev-parse --is-inside-work-tree *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Run this script inside a Git repository."
}

$findings = @()

if ($StagedOnly) {
    $findings += Test-Revision -Revision "staged index" -Cached
}
elseif ($History) {
    foreach ($commit in (& git rev-list --all)) {
        $findings += Test-Revision -Revision $commit
    }
}
else {
    $findings += Test-Revision -Revision "HEAD"
}

$findings = @($findings | Sort-Object -Unique)
if ($findings.Count -gt 0) {
    Write-Error ("Possible credential detected. Review and remove it:`n" + ($findings -join "`n"))
    exit 1
}

$scope = if ($StagedOnly) { "staged changes" } elseif ($History) { "full Git history" } else { "current HEAD" }
Write-Host "Secret scan passed: $scope"
exit 0
