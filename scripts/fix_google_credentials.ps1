# PowerShell script to properly format Google service account JSON for .env file
# Usage: .\fix_google_credentials.ps1 -JsonFilePath "config\google_credentials.json"

param(
    [Parameter(Mandatory=$true)]
    [string]$JsonFilePath
)

if (-not (Test-Path $JsonFilePath)) {
    Write-Host "Error: File not found: $JsonFilePath" -ForegroundColor Red
    exit 1
}

# Read JSON file
$jsonContent = Get-Content $JsonFilePath -Raw

# Validate JSON first
try {
    $jsonObject = $jsonContent | ConvertFrom-Json
    Write-Host "✓ JSON is valid" -ForegroundColor Green
} catch {
    Write-Host "Error: Invalid JSON file: $_" -ForegroundColor Red
    exit 1
}

# Convert to compact single-line JSON (properly escaped)
$compactJson = ($jsonObject | ConvertTo-Json -Compress -Depth 10)

# Escape backslashes and quotes for .env file
# In .env files, we need to escape quotes and backslashes
$escapedJson = $compactJson -replace '\\', '\\' -replace '"', '\"'

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Add this line to your .env file:" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "GOOGLE_SERVICE_ACCOUNT_JSON=$escapedJson" -ForegroundColor Yellow
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Or use base64 encoding (recommended for complex JSON):" -ForegroundColor Green
Write-Host ""

# Convert to base64
$bytes = [System.Text.Encoding]::UTF8.GetBytes($compactJson)
$base64 = [Convert]::ToBase64String($bytes)

Write-Host "GOOGLE_SERVICE_ACCOUNT_JSON_BASE64=$base64" -ForegroundColor Yellow
Write-Host ""
Write-Host "Note: If using base64, update utils.py to decode it automatically" -ForegroundColor Gray
Write-Host ""


