"""
Pydantic models for request/response
"""
from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask about Jamshed Ali")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are Jamshed's skills?"
            }
        }

class AnswerResponse(BaseModel):
    question: str
    answer: str
    
class HealthResponse(BaseModel):
    status: str
    message: str