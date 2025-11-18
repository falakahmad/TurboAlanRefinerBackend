# PowerShell script to convert Google service account JSON file to environment variable format
# Usage: .\convert_credentials_to_env.ps1 -Path "config\google_credentials.json"

param(
    [Parameter(Mandatory=$true)]
    [string]$Path
)

if (-not (Test-Path $Path)) {
    Write-Host "Error: File not found: $Path" -ForegroundColor Red
    exit 1
}

# Read JSON file
$jsonContent = Get-Content $Path -Raw

# Convert to single-line JSON (remove newlines and extra spaces)
$jsonContent = $jsonContent -replace '\s+', ' ' -replace '\s*{\s*', '{' -replace '\s*}\s*', '}' -replace '\s*,\s*', ',' -replace '\s*:\s*', ':'

# Escape double quotes for .env file
$escapedJson = $jsonContent -replace '"', '\"'

Write-Host ""
Write-Host "Add this to your .env file:" -ForegroundColor Green
Write-Host ""
Write-Host "GOOGLE_SERVICE_ACCOUNT_JSON=$escapedJson" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or set it in PowerShell:" -ForegroundColor Green
Write-Host '$env:GOOGLE_SERVICE_ACCOUNT_JSON = "' + $jsonContent + '"' -ForegroundColor Yellow
Write-Host ""


