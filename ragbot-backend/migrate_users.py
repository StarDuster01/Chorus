#!/usr/bin/env python3
"""
Migration script to move users.json from old location to new persistent location.
Run this once after updating the auth handlers.
"""

import os
import json
import shutil
from constants import USERS_FOLDER

def migrate_users():
    """Migrate users.json from old location to new persistent location"""
    
    # Old location (in handlers directory)
    old_users_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handlers", "users.json")
    
    # New location (in persistent storage)
    new_users_file = os.path.join(USERS_FOLDER, "users.json")
    
    print(f"Checking for existing users.json at: {old_users_file}")
    
    if os.path.exists(old_users_file):
        print(f"Found existing users.json, migrating to: {new_users_file}")
        
        # Ensure new directory exists
        os.makedirs(USERS_FOLDER, exist_ok=True)
        
        # Copy the file to new location
        shutil.copy2(old_users_file, new_users_file)
        
        # Verify migration
        if os.path.exists(new_users_file):
            with open(new_users_file, 'r') as f:
                users = json.load(f)
            print(f"✅ Migration successful! {len(users)} users migrated.")
            
            # Optionally remove old file (commented out for safety)
            # os.remove(old_users_file)
            print(f"⚠️  Old file still exists at {old_users_file} - you can manually delete it after verifying the migration worked.")
        else:
            print("❌ Migration failed - new file not found")
    else:
        print("No existing users.json found - nothing to migrate")
        
        # Ensure the users directory exists for future use
        os.makedirs(USERS_FOLDER, exist_ok=True)
        print(f"✅ Users directory created at: {USERS_FOLDER}")

if __name__ == "__main__":
    migrate_users() 