# RagBot

A powerful Retrieval-Augmented Generation (RAG) chatbot platform that allows you to create custom AI assistants with your own documents as the knowledge base.

## Features

- **Custom RAG Chatbots**: Create multiple bots backed by your own document collections
- **Multi-Document Support**: Upload and process PDFs, Word documents, PowerPoint, and text files
- **Model Chorus Technology**: Combine multiple LLMs for better responses
- **Visual Analysis**: Upload and analyze images in conversations
- **Multiple LLM Support**: Integration with OpenAI, Anthropic Claude, and Groq
- **Conversation Management**: Save and manage conversations with your bots
- **Image Generation**: Create images with OpenAI's gpt-image-1 model
- **Advanced RAG Pipeline**: Optimized document chunking and retrieval

## Prerequisites

- Docker Desktop installed on your system
- OpenAI, Anthropic, and/or Groq API keysq

## Quick Start with Docker

1. Clone this repository:
   ```
   git clone https://github.com/StarDuster01/RagBot.git
   cd RagBot
   ```

2. Create a `.env` file in the ragbot-backend directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   GROQ_API_KEY=your_groq_key_here
   JWT_SECRET=choose_a_secure_random_string
   ```

3. Start the application using Docker Compose:
   ```
   docker-compose up --build
   ```

4. Access the application:
   - Frontend: http://localhost
   - Backend API: http://localhost:50505

## Project Structure

- `ragbot-backend/`: Contains the Flask backend application
  - `Dockerfile`: Backend container configuration
  - `uploads/`: Directory where uploaded documents and images are stored
  - `chroma_db/`: Vector database storage
  - `datasets/`, `bots/`, `chorus/`, `conversations/`: Configuration and data storage
- `ragbot-frontend/`: Contains the React frontend application
  - `Dockerfile`: Frontend container configuration
  - `nginx.conf`: Nginx configuration for serving the frontend
- `docker-compose.yml`: Orchestrates the frontend and backend containers

## Docker Configuration

The application uses two containers:

1. **Backend Container**:
   - Runs on port 50505
   - Uses Python 3.11
   - Includes all necessary dependencies for document processing
   - Optimized for performance with 2 workers

2. **Frontend Container**:
   - Runs on port 80
   - Serves the React application
   - Uses Nginx for static file serving
   - Proxies API requests to the backend

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

The frontend is built with React and uses:
- Bootstrap for styling
- Axios for API communication
- React Router for navigation

## License

[Your license information here]

## Contact

[Your contact information here] 