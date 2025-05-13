# RAGBot Installation Guide

## Overview

RAGBot is a Flask-based application with a React frontend that provides a conversational AI interface with Retrieval-Augmented Generation capabilities. This guide will help you install and run the application on your system.

## System Requirements

- Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)
- 4GB RAM minimum (8GB+ recommended)
- 2GB free disk space
- Internet connection for API calls to OpenAI, Anthropic, and Groq

## Installation

### Option 1: Using the Installer (Windows)

1. Download the `RAGBot_Setup.exe` file
2. Run the installer and follow the on-screen instructions
3. The application will be installed to your Program Files directory
4. Launch RAGBot from the Start Menu or desktop shortcut

### Option 2: Manual Installation

1. Extract the `RAGBot.zip` archive to a location of your choice
2. Navigate to the extracted directory
3. Copy the `.env.template` file to `.env` and edit it with your API keys
4. Run `RAGBot.exe` (Windows) or `./RAGBot` (macOS/Linux)

## Configuration

Before using RAGBot, you need to configure your API keys:

1. Create accounts on OpenAI, Anthropic, and Groq (if you haven't already)
2. Generate API keys from each service
3. Open the `.env` file in a text editor
4. Replace the placeholder values with your actual API keys
5. Save the file

Example `.env` file:
```
OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456
ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456
GROQ_API_KEY=gsk_abcdefghijklmnopqrstuvwxyz123456
JWT_SECRET=my-strong-secret-key-for-jwt-tokens
```

## First-Time Setup

When you first run RAGBot, you'll need to:

1. Create a user account through the registration page
2. Create datasets and upload documents for the RAG functionality
3. Create a bot that uses your datasets

## Usage Guide

### Managing Datasets

1. Navigate to the Datasets tab
2. Click "New Dataset" to create a new dataset
3. Upload documents (PDF, DOCX, TXT) to your dataset
4. The documents will be processed and indexed for retrieval

### Creating Bots

1. Navigate to the Bots tab
2. Click "New Bot" to create a new bot
3. Configure the bot with a name, description, and system instructions
4. Select datasets for the bot to use
5. Optionally configure a Model Chorus for enhanced responses

### Chatting with Your Bot

1. Navigate to the Chat tab
2. Select your bot from the dropdown
3. Start a new conversation
4. Type your questions or upload images to interact with the bot

## Troubleshooting

### Common Issues

- **API Key Error**: Ensure your API keys are correctly entered in the `.env` file
- **Database Connection Error**: Check that the application has write permissions to its directory
- **Frontend Not Loading**: Try clearing your browser cache or using a different browser

### Logs

Logs are stored in the `logs` directory in the application folder. Include these logs when reporting issues.

## Support

If you encounter any issues with RAGBot, please contact support at support@example.com or open an issue on our GitHub repository.

## License

RAGBot is licensed under the terms specified in the LICENSE file included with the application.