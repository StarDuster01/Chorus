# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in this project, please report it by emailing the maintainers directly. **Do not create a public GitHub issue.**

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (if applicable)

We will respond to security reports within 48 hours.

---

## Security Best Practices for Deployment

### 1. Environment Variables (CRITICAL)

**Never commit API keys or secrets to the repository!**

All sensitive configuration must be set via environment variables:

```bash
# Required
OPENAI_API_KEY=your-key-here
JWT_SECRET=your-random-secret

# Optional but recommended
OPENAI_IMAGE_API_KEY=separate-image-key
ANTHROPIC_API_KEY=your-anthropic-key
GROQ_API_KEY=your-groq-key
```

Generate a secure JWT secret:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. User Credentials

**Do NOT commit `ragbot-backend/users.json`**

This file contains user credentials (even though hashed) and should:
- Be in `.gitignore` ✅ (already added)
- Never be committed to version control
- Be backed up securely outside the repository
- Use strong bcrypt hashing (already implemented)

### 3. Azure/Cloud Deployment

**Do NOT hardcode Azure credentials in deployment scripts**

Create a separate `deploy-config.sh` (gitignored) for sensitive deployment info:

```bash
# deploy-config.sh (add to .gitignore)
export AZURE_SUBSCRIPTION="your-subscription-name"
export AZURE_REGISTRY="your-registry.azurecr.io"
export AZURE_RESOURCE_GROUP="your-resource-group"
export CONTAINER_APP_NAME="your-app-name"
```

Then source it in your deployment script:
```bash
source deploy-config.sh
```

### 4. Docker Security

The `.dockerignore` files are configured to exclude:
- `.env` files
- User data (`users.json`)
- Uploaded files
- Database files
- Git history

**Verify before building:**
```bash
docker build --no-cache -t test-build .
docker run test-build ls -la  # Check what's included
```

### 5. API Rate Limiting

**Important:** This application does not currently implement rate limiting.

For production deployment, implement:
- Rate limiting on API endpoints
- Request throttling for expensive operations (image generation)
- IP-based blocking for abuse
- API key quotas

Recommended: Use a reverse proxy (nginx, Cloudflare) with rate limiting.

### 6. File Upload Security

The application accepts file uploads. Ensure:
- Files are scanned for malware
- File types are validated (already implemented)
- File sizes are limited (already implemented: 1000MB max)
- Uploaded files are stored outside the web root
- File names are sanitized (already implemented via `secure_filename`)

### 7. CORS Configuration

CORS is enabled for all origins. For production:

```python
# Restrict to specific origins
CORS(app, origins=["https://yourdomain.com"])
```

### 8. HTTPS/TLS

**Always use HTTPS in production!**

- Azure Container Apps: Enable automatic HTTPS
- Custom domains: Configure TLS certificates
- Never send JWT tokens over HTTP

---

## Known Security Considerations

### 1. Image Generation Costs

Image generation via OpenAI API can be expensive. Consider:
- Implementing usage quotas per user
- Adding cost alerts
- Rate limiting image generation requests
- Requiring authentication for all image operations

### 2. Data Persistence

User data is stored locally in JSON files:
- `users.json` - User credentials
- `conversations/` - Chat history
- `datasets/` - Uploaded documents
- `bots/` - Bot configurations

**For production:**
- Implement proper database (PostgreSQL, MongoDB)
- Encrypt sensitive data at rest
- Regular backups with encryption
- Implement data retention policies

### 3. Authentication Token Expiry

JWT tokens expire after 24 hours. Consider:
- Implementing refresh tokens
- Shorter expiry for sensitive operations
- Token revocation mechanism
- Multi-factor authentication for admin users

### 4. Input Validation

While basic validation is implemented, enhance for production:
- Stricter prompt filtering
- Content moderation for generated images
- SQL injection prevention (if moving to SQL database)
- XSS prevention in markdown rendering

---

## Security Checklist for Production

- [ ] All API keys in environment variables (not hardcoded)
- [ ] Strong JWT secret generated and set
- [ ] `users.json` not in version control
- [ ] HTTPS enabled
- [ ] CORS restricted to specific domains
- [ ] Rate limiting implemented
- [ ] File upload scanning enabled
- [ ] Security headers configured (X-Frame-Options, etc.)
- [ ] Error messages don't leak sensitive info
- [ ] Logging configured (without logging secrets)
- [ ] Regular dependency updates
- [ ] Security monitoring/alerts enabled

---

## Dependencies

Regularly update dependencies to patch security vulnerabilities:

```bash
# Backend
cd ragbot-backend
pip install --upgrade -r requirements.txt

# Frontend
cd ragbot-frontend
npm audit fix
npm update
```

Monitor for vulnerabilities:
- Backend: `pip-audit` or `safety`
- Frontend: `npm audit`

---

## Compliance Notes

### Data Privacy

This application may store:
- User credentials
- Uploaded documents
- Chat conversations
- Generated images

Ensure compliance with:
- GDPR (if serving EU users)
- CCPA (if serving California users)
- Your organization's data policies

Implement:
- Data deletion on user request
- Privacy policy
- Terms of service
- User consent for data storage

### Content Moderation

OpenAI's content policy is applied to:
- Generated images
- Chat responses

You are responsible for:
- Monitoring user-generated content
- Implementing additional filters if needed
- Compliance with local content laws

---

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OpenAI Safety Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [Azure Security Best Practices](https://docs.microsoft.com/en-us/azure/security/)

---

**Last Updated:** 2025-01-30  
**Version:** 2.0
