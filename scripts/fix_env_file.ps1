# Quick script to fix .env file for Google Drive credentials
# Usage: .\fix_env_file.ps1

$envFile = ".env"

if (-not (Test-Path $envFile)) {
    Write-Host "Error: .env file not found!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Fixing .env file for Google Drive credentials..." -ForegroundColor Cyan
Write-Host ""

# Read current .env file
$content = Get-Content $envFile -Raw

# Check if GOOGLE_SERVICE_ACCOUNT_JSON is set (even if empty)
if ($content -match "GOOGLE_SERVICE_ACCOUNT_JSON\s*=") {
    Write-Host "Found GOOGLE_SERVICE_ACCOUNT_JSON in .env file" -ForegroundColor Yellow
    
    # Comment out or remove the line
    $content = $content -replace "(?m)^(GOOGLE_SERVICE_ACCOUNT_JSON\s*=.*)$", "# `$1"
    Write-Host "  → Commented out GOOGLE_SERVICE_ACCOUNT_JSON" -ForegroundColor Green
}

# Check if GOOGLE_SERVICE_ACCOUNT_FILE is already set
if ($content -match "GOOGLE_SERVICE_ACCOUNT_FILE\s*=") {
    Write-Host "Found GOOGLE_SERVICE_ACCOUNT_FILE in .env file" -ForegroundColor Yellow
    
    # Update it to use the correct path
    $content = $content -replace "(?m)^(GOOGLE_SERVICE_ACCOUNT_FILE\s*=).*$", "`$1config/google_credentials.json"
    Write-Host "  → Updated GOOGLE_SERVICE_ACCOUNT_FILE to use config/google_credentials.json" -ForegroundColor Green
} else {
    # Add the line if it doesn't exist
    Write-Host "Adding GOOGLE_SERVICE_ACCOUNT_FILE to .env file" -ForegroundColor Yellow
    
    # Find a good place to add it (after Google Drive Configuration comment or at the end)
    if ($content -match "(?m)^(# Google Drive Configuration.*?)(\r?\n)") {
        # Add after Google Drive Configuration section
        $content = $content -replace "(?m)^(# Google Drive Configuration.*?)(\r?\n)", "`$1`$2GOOGLE_SERVICE_ACCOUNT_FILE=config/google_credentials.json`$2"
    } else {
        # Add at the end
        $content += "`nGOOGLE_SERVICE_ACCOUNT_FILE=config/google_credentials.json`n"
    }
    Write-Host "  → Added GOOGLE_SERVICE_ACCOUNT_FILE=config/google_credentials.json" -ForegroundColor Green
}

# Write back to file
$content | Set-Content $envFile -NoNewline

Write-Host ""
Write-Host "✅ .env file updated successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Restart your backend server" -ForegroundColor White
Write-Host "2. Try browsing Google Drive files again" -ForegroundColor White
Write-Host ""


