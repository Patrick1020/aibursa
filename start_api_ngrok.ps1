# start_api_ngrok.ps1
# 1) Pornește FastAPI (uvicorn)
# 2) Pornește ngrok (HTTPS)
# 3) Obține URL-ul public
# 4) Generează openapi.public.json corect (servers, operationId, schemas)

param(
  [string]$BindHost = "127.0.0.1",
  [int]$BindPort = 8000,
  [string]$App = "app.main:app",
  [string]$NgrokPath = "$env:USERPROFILE\OneDrive\Desktop\ngrok.lnk",
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Step 0: verific ngrok ==="
if (-not (Test-Path $NgrokPath)) {
  Write-Error "Nu găsesc ngrok la: $NgrokPath. Instalează de la https://ngrok.com/download sau schimbă calea."
  exit 1
}

# 1) FastAPI (uvicorn) pe localhost
Write-Host "=== Step 1: pornesc FastAPI (uvicorn) pe http://$($BindHost):$($BindPort) ==="
$uvicornCmd = "$Python -m uvicorn $App --host $BindHost --port $BindPort --reload"
Start-Process powershell -ArgumentList "-NoExit","-Command",$uvicornCmd
Start-Sleep -Seconds 2

# 2) ngrok http <port>
Write-Host "=== Step 2: pornesc ngrok (http $($BindPort)) ==="
$ngrokCmd = "`"$NgrokPath`" http $BindPort --log=stdout"
Start-Process powershell -ArgumentList "-NoExit","-Command",$ngrokCmd
Start-Sleep -Seconds 3

# 3) Ia public_url HTTPS din API-ul local ngrok (4040)
Write-Host "=== Step 3: obțin public_url de la ngrok ==="
$publicUrl = $null
for ($i=0; $i -lt 20; $i++) {
  try {
    $resp = Invoke-RestMethod -UseBasicParsing -Uri "http://127.0.0.1:4040/api/tunnels"
    $httpsTunnel = $resp.tunnels | Where-Object { $_.public_url -like "https://*" }
    if ($httpsTunnel) { $publicUrl = $httpsTunnel.public_url; break }
  } catch {}
  Start-Sleep -Seconds 1
}

if (-not $publicUrl) {
  Write-Error "Nu am putut obține public_url HTTPS de la ngrok (port 4040)."
  exit 1
}
Write-Host "ngrok public URL: $publicUrl"

# 4) Generează openapi.public.json cu servers, operationId etc.
Write-Host "=== Step 4: generez openapi.public.json (server=$publicUrl) ==="
$updCmd = "$Python .\tools\update_openapi.py --local http://$($BindHost):$($BindPort) --server $publicUrl --out openapi.public.json"
& powershell -Command $updCmd

if (-not (Test-Path ".\openapi.public.json")) {
  Write-Error "openapi.public.json nu a fost generat. Verifică erorile de mai sus."
  exit 1
}

# 4.1 Copiez schema in /static si afisez URL-ul public
if (-not (Test-Path ".\static")) {
    New-Item -ItemType Directory -Path ".\static" | Out-Null
}
Copy-Item ".\openapi.public.json" ".\static\openapi.public.json" -Force

# construiesc URL-ul public servit de FastAPI prin ngrok
$schemaUrl = "$publicUrl/static/openapi.public.json"
Write-Host "OpenAPI URL (public, pt. GPT Actions): $schemaUrl"



Write-Host ""
Write-Host "=== Gata! ==="
Write-Host "FastAPI local:    http://$($BindHost):$($BindPort)"
Write-Host "Public (ngrok):   $publicUrl"
Write-Host "OpenAPI public:   $(Resolve-Path .\openapi.public.json)"
Write-Host "OpenAPI public (served over ngrok): $schemaUrl"
Write-Host ""
Write-Host "1) Deschide $publicUrl/docs ca să verifici."
Write-Host "2) În GPT Actions, încarcă fișierul openapi.public.json."
