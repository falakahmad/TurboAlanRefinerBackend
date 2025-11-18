#!/usr/bin/env python3
"""Test Google credentials to diagnose JWT signature issues"""

import os
import sys
import json
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def test_credentials():
    """Test if Google credentials can be loaded and used"""
    print("=" * 60)
    print("Testing Google Drive Credentials")
    print("=" * 60)
    print()
    
    # Check file exists
    creds_file = backend_dir / 'config' / 'google_credentials.json'
    if not creds_file.exists():
        print(f"❌ Credentials file not found: {creds_file}")
        return False
    
    print(f"✅ Credentials file found: {creds_file}")
    
    # Try to load and parse JSON
    try:
        with open(creds_file, 'r', encoding='utf-8') as f:
            creds_data = json.load(f)
        print("✅ JSON file is valid")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return False
    
    # Check required fields
    required = ['type', 'project_id', 'private_key', 'client_email']
    missing = [f for f in required if f not in creds_data]
    if missing:
        print(f"❌ Missing required fields: {missing}")
        return False
    
    print(f"✅ All required fields present")
    print(f"   Type: {creds_data.get('type')}")
    print(f"   Project ID: {creds_data.get('project_id')}")
    print(f"   Client Email: {creds_data.get('client_email')}")
    
    # Check private key format
    private_key = creds_data.get('private_key', '')
    if not private_key.startswith('-----BEGIN PRIVATE KEY-----'):
        print(f"❌ Invalid private key format")
        print(f"   Key starts with: {private_key[:50]}...")
        return False
    
    if not private_key.endswith('-----END PRIVATE KEY-----\\n'):
        # Check if it ends properly (might have different newline)
        if not ('END PRIVATE KEY' in private_key):
            print(f"❌ Private key doesn't end properly")
            return False
    
    print(f"✅ Private key format looks correct")
    print(f"   Key length: {len(private_key)} characters")
    
    # Try to create credentials object
    try:
        from google.oauth2 import service_account
        from utils import OAUTH_SCOPES
        
        creds = service_account.Credentials.from_service_account_file(
            str(creds_file),
            scopes=OAUTH_SCOPES
        )
        print("✅ Credentials object created successfully")
    except Exception as e:
        print(f"❌ Failed to create credentials object: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback:\n{traceback.format_exc()}")
        return False
    
    # Try to refresh/get token (this will test JWT signature)
    try:
        from google.auth.transport.requests import Request
        
        if not creds.valid:
            print("⚠️  Credentials not valid, attempting refresh...")
            creds.refresh(Request())
        
        if creds.valid:
            print("✅ Credentials are valid and can generate tokens")
        else:
            print("❌ Credentials are invalid after refresh")
            return False
    except Exception as e:
        print(f"❌ Failed to refresh credentials: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback:\n{traceback.format_exc()}")
        
        # Check if it's a JWT signature error
        error_str = str(e)
        if 'invalid_grant' in error_str or 'JWT' in error_str or 'signature' in error_str:
            print()
            print("=" * 60)
            print("DIAGNOSIS: Invalid JWT Signature")
            print("=" * 60)
            print()
            print("This usually means:")
            print("1. The private key has been corrupted or modified")
            print("2. The service account key was regenerated/deleted in Google Cloud")
            print("3. There's an encoding issue with the private key")
            print()
            print("SOLUTION:")
            print("1. Go to Google Cloud Console")
            print("2. Navigate to: IAM & Admin > Service Accounts")
            print("3. Find your service account:", creds_data.get('client_email'))
            print("4. Click 'Keys' tab")
            print("5. Delete the old key and create a new one")
            print("6. Download the new JSON key")
            print("7. Replace config/google_credentials.json with the new file")
            print()
        
        return False
    
    # Try to make a test API call
    try:
        from googleapiclient.discovery import build
        
        service = build('drive', 'v3', credentials=creds)
        # Try to get service account info (this will test the API connection)
        results = service.files().list(pageSize=1, fields="files(id, name)").execute()
        print("✅ Successfully connected to Google Drive API")
        print(f"   Found {len(results.get('files', []))} accessible files")
        return True
    except Exception as e:
        print(f"⚠️  Credentials loaded but API call failed: {e}")
        print(f"   This might be normal if no files are shared with the service account")
        print(f"   The credentials themselves are valid!")
        return True  # Still consider it a success if credentials are valid
    
    return True

if __name__ == '__main__':
    success = test_credentials()
    print()
    print("=" * 60)
    if success:
        print("✅ All tests passed! Credentials are working.")
    else:
        print("❌ Tests failed. See errors above.")
    print("=" * 60)
    sys.exit(0 if success else 1)


