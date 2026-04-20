"""
FastAPI application for Jamshed Ali's Portfolio RAG Assistant
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.rag_assistant import PortfolioRAGAssistant
from app.models import QuestionRequest, AnswerResponse, HealthResponse
from app.config import settings

# Global assistant instance
assistant = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG assistant on startup"""
    global assistant
    print("🚀 Starting up...")
    assistant = PortfolioRAGAssistant(settings.PDF_PATH)
    assistant.initialize()
    yield
    print("🛑 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Jamshed Ali Portfolio Assistant API",
    description="RAG-based AI assistant to answer questions about Jamshed Ali",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {
        "status": "success",
        "message": "Jamshed Ali Portfolio Assistant API is running!"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "RAG Assistant is initialized and ready"
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about Jamshed Ali
    
    - **question**: Your question about Jamshed Ali's skills, projects, experience, etc.
    - **session_id**: Unique session ID to maintain conversation memory.
    """
    try:
        answer = assistant.ask(request.question, request.session_id)
        return {
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_questions():
    """
    Test endpoint with sample questions
    """
    test_cases = [
        "What are Jamshed's skills?",
        "Tell me about his projects",
        "What is his email?"
    ]
    
    results = []
    for question in test_cases:
        try:
            answer = assistant.ask(question)
            results.append({
                "question": question,
                "answer": answer
            })
        except Exception as e:
            results.append({
                "question": question,
                "answer": f"Error: {str(e)}"
            })
    
    return {"test_results": results}