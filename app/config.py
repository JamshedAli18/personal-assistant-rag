"""
Configuration file for environment variables
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "jamshed-portfolio-assistant")
    
    # PDF Path
    PDF_PATH = "info.pdf"

settings = Settings()