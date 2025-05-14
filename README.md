# RagBot

A powerful Retrieval-Augmented Generation (RAG) chatbot platform that allows you to create custom AI assistants with your own documents as the knowledge base.

## Features

- **Custom RAG Chatbots**: Create multiple bots backed by your own document collections
- **Multi-Document Support**: Upload and process PDFs, Word documents, PowerPoint, and text files
- **Model Chorus Technology**: Combine multiple LLMs for better responses
- **Visual Analysis**: Upload and analyze images in conversations
- **Multiple LLM Support**: Integration with OpenAI, Anthropic Claude, and Groq
- **Conversation Management**: Save and manage conversations with your bots
- **Image Generation**: Create images with OpenAI's DALL-E model
- **Advanced RAG Pipeline**: Optimized document chunking and retrieval

## Prerequisites

- Python 3.8+
- Node.js and npm (for frontend)
- OpenAI, Anthropic, and/or Groq API keys

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd RagBot
   ```

2. Set up the backend:
   ```
   cd ragbot-backend
   
   # Create virtual environment inside the backend directory
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   
   # Install required packages using the existing requirements.txt
   pip install -r requirements.txt
   ```

3. Set up the frontend:
   ```
   cd ../ragbot-frontend
   npm install
   ```

4. Create a `.env` file in the ragbot-backend directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   GROQ_API_KEY=your_groq_key_here
   JWT_SECRET=choose_a_secure_random_string
   ```

## Project Structure

- `ragbot-backend/`: Contains the Flask backend application
  - `app.py`: Main application file
  - `uploads/`: Directory where uploaded documents and images are stored
  - `chroma_db/`: Vector database storage
  - `datasets/`, `bots/`, `chorus/`, `conversations/`: Configuration and data storage
  - `requirements.txt`: Python dependencies
  - `venv/`: Virtual environment (to be created during installation)
- `ragbot-frontend/`: Contains the React frontend application
- `start_ragbot.bat`: Windows batch script to start both backend and frontend

## Running the Application

### Method 1: Using the Startup Script (Windows)

For Windows users, simply run the included batch script:
```
start_ragbot.bat
```

This script will:
1. Determine your local IP address
2. Start the backend server on port 5000
3. Start the frontend server on port 3000
4. Display URLs that can be used to access the application, including from other devices on your network

### Method 2: Manual Startup

1. Start the backend server:
   ```
   cd ragbot-backend
   venv\Scripts\activate  # On Windows
   # OR
   source venv/bin/activate  # On macOS/Linux
   python app.py
   ```
   The API server will run on http://localhost:5000

2. Start the frontend server (in a new terminal):
   ```
   cd ragbot-frontend
   npm start
   ```
   The frontend will run on http://localhost:3000

3. Access the application in your web browser at http://localhost:3000

## Usage

### Creating a Dataset

1. Register or log in to your account
2. Navigate to the Datasets section
3. Create a new dataset with a name and description
4. Upload documents to your dataset

### Creating a Bot

1. Go to the Bots section
2. Create a new bot with a name and description
3. Select one or more datasets to use as the knowledge base
4. Customize the system instructions for your bot

### Advanced: Setting up a Model Chorus

1. Go to the Chorus section
2. Create a new chorus configuration
3. Add response models (from OpenAI, Anthropic, or Groq)
4. Add evaluator models that will vote on the best responses
5. Assign the chorus to a bot

### Chatting with Your Bot

1. Select a bot from your list
2. Start a new conversation or continue an existing one
3. Type a question or upload an image
4. The bot will respond based on the content in your datasets

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `GROQ_API_KEY`: Your Groq API key
- `JWT_SECRET`: Secret key for JWT token generation

## Development

The backend is built with Flask and uses:
- ChromaDB for vector embeddings and semantic search
- OpenAI, Anthropic, and Groq for LLM capabilities
- JWT for authentication
- Various libraries for document processing

The frontend is built with React.

## License

[Your license information here]

## Contact

[Your contact information here] 