$ErrorActionPreference = 'Stop'

$bundle = Split-Path -Parent $MyInvocation.MyCommand.Path
$evidence = Join-Path $bundle 'results/evidence'
$manifestPath = Join-Path $evidence 'manifest.json'
$expectedManifest = 'adef14b5108e041aee54603a7680855c9306fab48ed03b2a50f827552babd95a'
$actualManifest = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
if ($actualManifest -ne $expectedManifest) { throw "Manifest SHA mismatch: $actualManifest" }

$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$count = 0
foreach ($entry in $manifest.files.PSObject.Properties) {
  $relative = $entry.Name.Replace('/', [IO.Path]::DirectorySeparatorChar)
  $path = Join-Path $evidence $relative
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing evidence: $relative" }
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
  if ($actual -ne $entry.Value.sha256) { throw "SHA mismatch: $relative" }
  if ((Get-Item -LiteralPath $path).Length -ne $entry.Value.bytes) { throw "Size mismatch: $relative" }
  $count += 1
}
if ($count -ne 20) { throw "Unexpected evidence file count in manifest: $count" }
Write-Host "Evidence OK: manifest + $count payload files"
