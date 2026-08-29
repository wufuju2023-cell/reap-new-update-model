$ErrorActionPreference = 'Stop'

$bundle = Split-Path -Parent $MyInvocation.MyCommand.Path
$checks = @(
  @{ Path = 'code/train_value_head.py'; Sha = '43a7da1f3d4ea9f5c470a4daea7a472ea7e7450324b1f587a93cbae078cbd5cc' },
  @{ Path = 'code/categorical_search_backend.py'; Sha = '01124bf6ca1aae904bef1da377489267bba1075a17191f6a66d3f547d774a0cd' },
  @{ Path = 'code/online_ttt.py'; Sha = '0965f6c4a7aba001e879bfeb0bbcc3d9e27ca73c89aa3fb9f5abdfe13f410eb5' },
  @{ Path = 'code/success_learn_recovery.py'; Sha = '32e64a52a92cfb369a5e3701d205025bba82e6a782459f0a05924791ddc2fa0f' },
  @{ Path = 'code/test_success_learn_recovery.py'; Sha = '08bc0511a9605592ee033f799590d7eab9bfe1fb4403fe154bb8cd4ea81c0604' },
  @{ Path = 'config/ExecutionDeclarations.lean'; Sha = '6a27d4d7f9e1ae017fa998b95598396001530968844266e81dfde4f5805d3bdf' },
  @{ Path = 'results/full-v3-training-report.json'; Sha = 'f4f0be3947fe791f06fcfbd4a32c62b42f806e9c33269bf1d87e72f4e1bc0b54' }
)

foreach ($check in $checks) {
  $path = Join-Path $bundle $check.Path
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Missing bundle file: $($check.Path)"
  }
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
  if ($actual -ne $check.Sha) {
    throw "SHA mismatch for $($check.Path): $actual"
  }
  Write-Host "OK $($check.Path) $actual"
}

$report = Get-Content -Raw -LiteralPath (Join-Path $bundle 'results/full-v3-training-report.json') | ConvertFrom-Json
if ($report.architecture -ne 'linear-3584-silu-256-linear-64') { throw 'Unexpected architecture' }
if ($report.train_rows -ne 205628 -or $report.validation_rows -ne 8000 -or $report.test_rows -ne 8000) { throw 'Unexpected split sizes' }
if ($report.checkpoint_sha256 -ne '7d6ab63f355013f6821b41bf20c31982e8e0bc7baad5d91243168a1ff36fdbc2') { throw 'Unexpected checkpoint identity' }
Write-Host 'Bundle report contract OK'
