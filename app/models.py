"""
Pydantic models for request/response
"""
from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask about Jamshed Ali")
    session_id: str = Field(default="default_session", description="Unique session ID for conversation memory")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are Jamshed's skills?",
                "session_id": "user_123"
            }
        }

class AnswerResponse(BaseModel):
    question: str
    answer: str
    
class HealthResponse(BaseModel):
    status: str
    message: str