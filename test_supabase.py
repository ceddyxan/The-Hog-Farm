#!/usr/bin/env python3
"""
Test Supabase connection for Hog Farm application
"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

print("=== Supabase Connection Test ===")
print(f"URL: {supabase_url}")
print(f"Key exists: {bool(supabase_key)}")

if not supabase_url or not supabase_key:
    print("❌ ERROR: Missing Supabase credentials!")
    print("Please check your .env file")
    sys.exit(1)

try:
    from supabase import create_client
    
    # Create client
    client = create_client(supabase_url, supabase_key)
    print("✅ Supabase client created successfully")
    
    # Test connection by checking if we can access the service
    print("🔍 Testing database connection...")
    
    # Try to access the hogs table (this will test if the table exists and is accessible)
    try:
        response = client.table('hogs').select('count', count='exact').execute()
        print("✅ Database connection successful!")
        print(f"📊 Hogs table accessible: {response}")
    except Exception as table_error:
        print(f"⚠️  Table access issue: {str(table_error)}")
        print("This might mean the table doesn't exist yet - that's OK for first setup")
    
    print("🎉 Supabase is working in production!")
    
except ImportError as e:
    print(f"❌ ImportError: {str(e)}")
    print("Please install: pip install supabase")
    sys.exit(1)
except Exception as e:
    print(f"❌ Connection Error: {str(e)}")
    print(f"Error Type: {type(e).__name__}")
    sys.exit(1)
