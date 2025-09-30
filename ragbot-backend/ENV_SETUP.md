# Environment Variables Setup

## Required Environment Variables

Create a `.env` file in the `ragbot-backend` directory with the following variables:

### OpenAI API Keys
```bash
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_IMAGE_API_KEY=your-openai-image-generation-key-here
```

**Note:** 
- `OPENAI_API_KEY` is used for general LLM tasks (chat, embeddings, etc.)
- `OPENAI_IMAGE_API_KEY` is specifically for image generation/editing
- If `OPENAI_IMAGE_API_KEY` is not set, it will fall back to `OPENAI_API_KEY`

### Anthropic API Key (for Claude models)
```bash
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

Required if you want to use Anthropic/Claude models in Model Chorus.

### Groq API Key (for Groq models)
```bash
GROQ_API_KEY=your-groq-api-key-here
```

Required if you want to use Groq models in Model Chorus.

### JWT Secret (for authentication) **REQUIRED**
```bash
JWT_SECRET=your-random-secret-key-here
```

**The application will NOT start without this!**

Generate a random secret for production:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Optional Variables

```bash
# External data directory (for Azure VM mounts, etc.)
EXTERNAL_DATA_DIR=/path/to/external/data

# Port (defaults to 50506 if not set)
PORT=50506
```

## Setup Instructions

1. Copy this template to create your `.env` file:
   ```bash
   cd ragbot-backend
   touch .env
   ```

2. Edit `.env` and add your API keys

3. **Important:** Never commit `.env` to git! It should already be in `.gitignore`

4. For Docker deployment, make sure `.env` is in the same directory as `docker-compose.yml`

## Verifying Setup

After setting up your `.env` file, the backend will:
- Print warnings on startup if required API keys are missing
- Fall back gracefully where possible (e.g., image generation uses main OpenAI key if separate key not set)

## Security Notes

- ✅ All API keys are now loaded from environment variables
- ✅ No API keys are hardcoded in the source code
- ✅ `.env` file should be in `.gitignore`
- ⚠️ Never share or commit API keys to version control
- ⚠️ Rotate keys if accidentally exposed
