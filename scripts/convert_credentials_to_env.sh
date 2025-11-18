#!/bin/bash
# Helper script to convert Google service account JSON file to environment variable format
# Usage: ./convert_credentials_to_env.sh path/to/google_credentials.json

if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_google_credentials.json>"
    echo "Example: $0 config/google_credentials.json"
    exit 1
fi

CREDENTIALS_FILE="$1"

if [ ! -f "$CREDENTIALS_FILE" ]; then
    echo "Error: File not found: $CREDENTIALS_FILE"
    exit 1
fi

# Convert JSON to single-line format and escape quotes
CREDENTIALS=$(cat "$CREDENTIALS_FILE" | jq -c . 2>/dev/null || cat "$CREDENTIALS_FILE" | tr -d '\n' | sed 's/"/\\"/g')

if [ -z "$CREDENTIALS" ]; then
    echo "Error: Failed to read credentials file"
    exit 1
fi

echo ""
echo "Add this to your .env file:"
echo ""
echo "GOOGLE_SERVICE_ACCOUNT_JSON=$CREDENTIALS"
echo ""
echo "Or export it directly:"
echo "export GOOGLE_SERVICE_ACCOUNT_JSON='$CREDENTIALS'"
echo ""


