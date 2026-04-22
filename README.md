# Jamshed Ali Portfolio Assistant

A RAG-based AI assistant to answer questions about Jamshed Ali's portfolio, skills, and projects.

## Features
- LangChain + LangGraph expertise
- RAG pipeline with Pinecone vector database
- Groq LLM (Llama 3.1 8B)
- Cohere embeddings
- FastAPI REST API

## API Endpoints

### `GET /`
Health check

### `GET /health`
Service status

### `POST /ask`
Ask a question about me


### `GET /test`
Run test questions

## Environment Variables
- `GROQ_API_KEY`
- `PINECONE_API_KEY`
- `COHERE_API_KEY`
- `PINECONE_INDEX_NAME`

## Local Development

```bash
# Install dependencies
uv sync

# Run server
uv run uvicorn main:app --reload
```

## Deployment
Deployed on Render: [Your Render URL]
