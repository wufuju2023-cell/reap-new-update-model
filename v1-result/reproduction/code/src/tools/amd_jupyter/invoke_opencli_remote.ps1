param(
    [Parameter(Mandatory=$true)][string]$Profile,
    [Parameter(Mandatory=$true)][string]$CommandFile,
    [string]$Session = 'ms',
    [string]$SaveOutput
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$cli = Join-Path $repo '.downloads/opencli-tooling/node_modules/@jackwener/opencli/dist/src/main.js'
$payload = @{command=[System.IO.File]::ReadAllText((Resolve-Path $CommandFile).Path)} | ConvertTo-Json -Compress
# Discover only the command endpoint observed in the current IDE frame.
# Do not read cookies, local storage, token responses, or historical gateways.
$template = @'
(async()=>{
  if(location.origin!=='https://www.modelscope.cn') throw new Error('Wrong page origin');
  if(!document.body.innerText.includes('DSW-AMD')) throw new Error('Current page is not AMD workspace');
  const frame=document.querySelector('iframe[title="notebook-ide"]');
  if(!frame) throw new Error('No current IDE frame');
  const urls=[...new Set(frame.contentWindow.performance.getEntriesByType('resource')
    .map(e=>new URL(e.name)).filter(u=>u.hostname==='dsw-gateway-cn-hangzhou.data.aliyun.com' && /^\/dsw-\d+\/dsw\/commands$/.test(u.pathname))
    .map(u=>u.origin+u.pathname))];
  if(urls.length!==1) throw new Error('Missing or ambiguous current gateway');
  const response=await fetch(urls[0]+'?type=status&name=next-ide',{
    method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},
    signal:AbortSignal.timeout(45000),body:JSON.stringify(__PAYLOAD__)
  });
  return {status:response.status,text:await response.text()};
})()
'@
$js = $template.Replace('__PAYLOAD__', $payload)
$raw = & node $cli --profile $Profile browser $Session eval $js
if ($LASTEXITCODE -ne 0) { throw 'OpenCLI command failed; do not automatically retry a mutation' }
$envelope = ($raw -join "`n") | ConvertFrom-Json
if ($envelope.status -ne 200) { throw "Remote HTTP failure: $($envelope.status)" }
$body = $envelope.text | ConvertFrom-Json
if ($null -eq $body.output) { throw 'Remote response has no output' }
if ($SaveOutput) {
    [System.IO.File]::WriteAllText($SaveOutput, $body.output+"`n", [System.Text.UTF8Encoding]::new($false))
}
Write-Output $body.output
