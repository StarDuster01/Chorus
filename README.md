# RAGBot - AI Assistant with Retrieval-Augmented Generation

RAGBot is a powerful AI assistant application that combines a Flask backend with a React frontend to provide conversational AI with document retrieval capabilities.

## Features

- **Retrieval-Augmented Generation**: Chat with AI that has access to your documents
- **Multi-Model Support**: Use OpenAI, Anthropic, and Groq models
- **Model Chorus**: Combine multiple models for enhanced responses
- **Document Processing**: Support for PDF, DOCX, TXT, and other document formats
- **Conversational Memory**: Store and retrieve conversation history
- **Image Analysis**: Upload and analyze images with AI
- **User Authentication**: Secure login and registration system

## Quick Start

### Windows

1. Run the installer `RAGBot_Setup.exe`
2. Launch the application from the desktop shortcut
3. Configure your API keys in the `.env` file
4. Open a browser to http://localhost:5000

### macOS/Linux

1. Extract the `RAGBot.zip` archive
2. Run `chmod +x start_ragbot.sh` to make the script executable
3. Configure your API keys in the `.env` file
4. Run `./start_ragbot.sh` to start the application
5. Open a browser to http://localhost:5000

## Development

If you want to modify the application:

1. Clone the repository
2. Install dependencies:
   - Backend: `pip install -r requirements.txt`
   - Frontend: `cd ragbot-frontend && npm install`
3. Run the backend: `python app.py`
4. Run the frontend: `cd ragbot-frontend && npm start`

## Packaging

To create a distributable package:

1. Run `python package_app.py`
2. Find the executable in the `dist/RAGBot` directory
3. Find the installer in the `installer` directory (Windows only)

## Documentation

For detailed documentation, see:

- [Installation Guide](INSTALLATION.md)
- [User Manual](MANUAL.md)
- [API Documentation](API.md)

## License

RAGBot is released under the MIT License. See [LICENSE](LICENSE) for details.