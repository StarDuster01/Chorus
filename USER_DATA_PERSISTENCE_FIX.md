# User Account Persistence Fix

## Problem Identified
Login accounts weren't being saved between redeployments because the `users.json` file was stored inside the Docker container's filesystem, which gets wiped out on every redeployment.

## Root Cause
- `users.json` was stored at `ragbot-backend/handlers/users.json` inside the container
- This location was not mounted as a persistent volume
- Docker container rebuilds destroyed user data on every deployment

## Solution Implemented

### 1. Updated Storage Structure
- Added `USERS_FOLDER` constant to `constants.py` pointing to `/ChorusAllData/users`
- Added `BOTS_FOLDER` constant for consistency (bots were already persistent via docker volume)
- Updated directory creation to include users folder

### 2. Updated Authentication Handlers
- Modified `ragbot-backend/handlers/auth_handlers.py` to use `USERS_FOLDER` from constants
- Both `register_handler` and `login_handler` now store users.json in persistent storage
- Removed redundant directory creation (handled in constants.py)

### 3. Created Migration Script
- Added `ragbot-backend/migrate_users.py` to help migrate existing users
- Script safely copies users from old location to new persistent location

## File Structure After Fix

```
ChorusAllData/                    # Persistent external storage
├── conversations/               # ✅ Already persistent
├── datasets/                    # ✅ Already persistent  
├── image_indices/              # ✅ Already persistent
├── uploads/                    # ✅ Already persistent
│   ├── images/
│   └── documents/
├── users/                      # 🆕 New - for user accounts
│   └── users.json             # 🆕 Now persistent
└── bots/                       # 🆕 Added to constants for consistency
    └── {user_id}_bots.json    # ✅ Already persistent via docker volume
```

## Next Steps

1. **Deploy the changes**:
   ```bash
   docker-compose down
   docker-compose up --build -d
   ```

2. **Run migration script** (if you have existing users):
   ```bash
   docker exec -it ragbot-backend-1 python3 migrate_users.py
   ```

3. **Verify the fix**:
   - Create a test user account
   - Redeploy the application
   - Confirm the user account still exists

## Verification
After deployment, you should see:
- `/ChorusAllData/users/` directory created
- New user registrations stored in `/ChorusAllData/users/users.json`
- User accounts persist through redeployments

## Optional Future Improvements
- Consider migrating to a proper database (PostgreSQL, SQLite) for better user management
- Add user management admin interface
- Implement user roles and permissions
- Add user profile management features 