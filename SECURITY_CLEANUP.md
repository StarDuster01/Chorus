# Security Cleanup Guide

## Before Making the Repository Public

This guide helps you remove sensitive data from your git repository before making it public.

## Step 1: Check for Sensitive Files

Run these commands to see what sensitive files are currently tracked:

```bash
# Check if sensitive files are in git
git ls-files | grep -E "(users\.json|\.env|deploy-config)"
```

## Step 2: Remove Sensitive Files from Git (But Keep Locally)

**Important:** These commands remove files from git history but keep them on your filesystem.

```bash
# Remove users.json from git (keeps local copy)
git rm --cached ragbot-backend/users.json

# Remove any .env files if accidentally committed
git rm --cached ragbot-backend/.env 2>/dev/null || true

# Remove any deploy config files if committed
git rm --cached deploy-config.sh 2>/dev/null || true
git rm --cached deploy-config.ps1 2>/dev/null || true
```

## Step 3: Verify .gitignore is Updated

The `.gitignore` file should already be updated (we just did this), but verify it contains:

```
# Security-sensitive files (DO NOT COMMIT)
ragbot-backend/users.json
ragbot-backend/image_indices/
deploy-config.sh
deploy-config.ps1
```

## Step 4: Commit the Changes

```bash
git add .gitignore
git commit -m "Security: Remove sensitive files and update gitignore"
```

## Step 5: Search for Any Remaining Secrets

### On Linux/Mac/WSL:
```bash
# Search for OpenAI API key patterns
grep -r "sk-" --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=venv .

# Search for potential API keys
grep -r "api.*key.*=" --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=venv .
```

### On Windows PowerShell:
```powershell
# Search for OpenAI API key patterns
Select-String -Path . -Pattern "sk-" -Recurse -Exclude *.git*,*node_modules*,*venv*

# Search for potential API keys  
Select-String -Path . -Pattern "api.*key.*=" -Recurse -Exclude *.git*,*node_modules*,*venv*
```

**Expected result:** Should find NO matches (all keys should be in `.env` now)

## Step 6: Clean Git History (OPTIONAL - Advanced)

If you previously committed secrets to git and want to remove them from history:

⚠️ **WARNING:** This rewrites git history and will affect anyone who has cloned the repo!

```bash
# Use BFG Repo-Cleaner (recommended)
# Download from: https://rtyley.github.io/bfg-repo-cleaner/

# Remove any file named users.json from entire history
bfg --delete-files users.json

# Remove any file containing API keys (create patterns.txt first)
bfg --replace-text patterns.txt

# Force push to update remote (use with caution!)
git push --force
```

### Alternative: git filter-branch (more complex)
```bash
# Remove users.json from entire history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch ragbot-backend/users.json" \
  --prune-empty --tag-name-filter cat -- --all

# Force garbage collection
git gc --aggressive --prune=now

# Force push
git push --force
```

## Step 7: Final Security Checklist

Before making the repo public, verify:

- [ ] No files with `users.json` in git: `git ls-files | grep users.json` (should be empty)
- [ ] No `.env` files in git: `git ls-files | grep .env` (should be empty)
- [ ] `.gitignore` includes all sensitive file patterns
- [ ] README.md contains security warning and links to SECURITY.md
- [ ] SECURITY.md exists with security guidelines
- [ ] ENV_SETUP.md exists with environment variable instructions
- [ ] All API keys are in environment variables (none hardcoded)
- [ ] JWT_SECRET requirement is enforced in app.py
- [ ] `deploy-config.example.sh` and `.ps1` exist as templates

## Step 8: Test Locally Before Going Public

```bash
# Create fresh clone to simulate new user
cd /tmp
git clone /path/to/your/repo ragbot-test
cd ragbot-test

# Verify sensitive files are NOT present
ls ragbot-backend/users.json  # Should not exist
ls ragbot-backend/.env         # Should not exist

# Try to run (should fail with helpful error about JWT_SECRET)
cd ragbot-backend
python app.py  # Should exit with JWT_SECRET error message
```

## Step 9: Create Public Repository

### Option 1: Push to New Public Repo
```bash
# Create new repo on GitHub/GitLab (set as public)
git remote add public https://github.com/yourusername/ragbot.git
git push public main
```

### Option 2: Make Existing Repo Public
1. Go to repository settings
2. Change visibility to "Public"
3. Confirm the change

## Step 10: Post-Publication Monitoring

After making the repo public:

1. **Monitor for accidental commits:**
   ```bash
   # Set up pre-commit hook to prevent committing secrets
   # Create .git/hooks/pre-commit
   ```

2. **Enable GitHub Secret Scanning** (if using GitHub):
   - Go to Settings → Security → Secret scanning
   - Enable both "Secret scanning" and "Push protection"

3. **Watch for issues:**
   - Monitor GitHub issues for security reports
   - Respond to security concerns promptly

## What to Do If Secrets Are Exposed

If you accidentally expose API keys or secrets:

1. **Immediately rotate the exposed keys:**
   - OpenAI: https://platform.openai.com/api-keys
   - Anthropic: https://console.anthropic.com/
   - Groq: https://console.groq.com/

2. **Update all deployments** with new keys

3. **Follow git history cleaning steps above**

4. **Monitor for unauthorized usage** of exposed keys

---

## Quick Commands Summary

```bash
# Remove sensitive files from git
git rm --cached ragbot-backend/users.json
git rm --cached ragbot-backend/.env 2>/dev/null || true

# Commit changes
git add .gitignore
git commit -m "Security: Remove sensitive files and update gitignore"

# Verify no sensitive files remain
git ls-files | grep -E "(users\.json|\.env)"

# Push changes
git push origin main
```

---

**Remember:** Security is an ongoing process, not a one-time task. Regularly review your repository for sensitive data and keep dependencies updated!
