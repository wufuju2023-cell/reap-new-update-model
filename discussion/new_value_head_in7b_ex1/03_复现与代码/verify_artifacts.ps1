$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$checks = @(
    @{ Path = '.downloads/new-value-head-20260829/train205628-full-v3/value-head.pt'; Sha256 = '7d6ab63f355013f6821b41bf20c31982e8e0bc7baad5d91243168a1ff36fdbc2' },
    @{ Path = '.downloads/new-value-head-20260829/train205628-full-v3/value-head-initial.pt'; Sha256 = 'e0cab9659a89765b4d64aee8d655bc9ea921eca181772d7a3cd215bb4de9cec7' },
    @{ Path = '.downloads/new-value-head-20260829/radeon-fullv3-handoff/handoff-manifest.json'; Sha256 = 'b382e5e60a75373b7d901f860697cc585717fb0d2367f51369e351a3b681860f' }
)

foreach ($check in $checks) {
    $path = Join-Path $repo $check.Path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing artifact: $($check.Path)"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $check.Sha256) {
        throw "SHA256 mismatch: $($check.Path) expected=$($check.Sha256) actual=$actual"
    }
    [pscustomobject]@{ Path = $check.Path; SHA256 = $actual; Status = 'ok' }
}
