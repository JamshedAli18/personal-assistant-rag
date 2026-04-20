"""
Personal Portfolio RAG Assistant for Jamshed Ali
"""
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from pinecone import Pinecone, ServerlessSpec
from .config import settings


class PortfolioRAGAssistant:
    """RAG Assistant for Jamshed Ali's Portfolio"""
    
    def __init__(self, pdf_path: str):
        """Initialize the RAG assistant"""
        self.pdf_path = pdf_path
        
        # Initialize LLM with better settings
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.4,
            max_tokens=300,
        )
        
        # Initialize embeddings
        self.embeddings = CohereEmbeddings(
            cohere_api_key=settings.COHERE_API_KEY,
            model="embed-english-v3.0"
        )
        
        self.vector_store = None
        self.qa_chain = None
        
        # Simple session-based memory store
        self.chat_histories = {}
        
    def load_and_chunk_pdf(self):
        """Load PDF and split into chunks"""
        print("[PDF] Loading PDF...")
        
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        print(f"OK: Loaded {len(documents)} pages from PDF")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"OK: Created {len(chunks)} chunks")
        
        return chunks
        
    def setup_pinecone(self, pc):
        """Initialize Pinecone index"""
        print("[Pinecone] Setting up index...")
        
        if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print(f"Creating new index: {settings.PINECONE_INDEX_NAME}")
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("✅ Index created successfully")
        else:
            print(f"OK: Using existing index: {settings.PINECONE_INDEX_NAME}")
            
    def create_vector_store(self, chunks):
        """Create vector store from chunks"""
        print("[VectorStore] Creating vector store...")
        
        self.vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            index_name=settings.PINECONE_INDEX_NAME
        )
        
        print("OK: Vector store created successfully")
        
    def setup_qa_chain(self):
        """Setup the QA chain with improved custom prompt"""
        print("[QAChain] Setting up QA chain...")
        
        template = """You are the personal assistant of Jamshed Ali, a professional AI Engineer from Karachi, Pakistan. Use the following context from his resume to answer questions.

### CONTEXT:
{context}

### CHAT HISTORY:
{chat_history}

### RESPONSE RULES:

1. **Greetings (Hi, Hello, How are you, What's up, etc.):**
   - Respond warmly and naturally with variety. Never repeat the same greeting or response.
   - Use ONE funny/friendly emoji at the end
   - Examples: 
     * "I'm doing great! How can I help you learn about Jamshed today? 👋"
     * "Hey there! Ready to explore Jamshed's work? 🚀"
     * "Hello! I'm here to help. What would you like to know? 😊"
     * "Doing well! What questions do you have about Jamshed? ✨"
   - **CRITICAL:** Emojis are ONLY for greetings, never in other responses

2. **Identity Questions (Who are you, What are you, etc.):**
   - If asked specifically about YOU (the assistant): "I'm Jamshed's personal assistant. He integrated me to help answer questions about his work and expertise."
   - If asked about Jamshed (Who is Jamshed, Who is he, etc.): Provide a concise 1-sentence professional summary (Name, title, location). Do NOT include project details here.

3. **Pronoun Understanding:**
   - "he", "his", "him" = Jamshed Ali
   - Example: "What are his skills?" = "What are Jamshed's skills?"

4. **Contact Information:**
   - Jump DIRECTLY to contact details, no preamble
   - Include: Email, Phone, Location, LinkedIn, GitHub
   - Format:
     Email: [email]
     Phone: [number]
     Location: [city]
     LinkedIn: [link]
     GitHub: [link]

5. **Certifications:**
   - List all certifications with issuing organization and year
   - Be complete and clear
   - No preamble, jump straight to the list
   - Format: "Certification Name - Organization (Year)"

6. **Training & Experience:**
   - Jump straight to the details
   - Mention the HEC training program, duration, and "Top Performer" achievement
   - No introductory phrases

7. **Projects:**
    - When asked "what projects", "tell about projects", "what systems he built", or "worked with", ALWAYS list ALL of these projects:
      * AI Legal Document Analyzer (using LangGraph)
      * Tutorix (LLMATES) — AI-Powered Tutor Chatbot
      * AI SQL Agent (TechCorp DB)
      * AI Job Application Assistant
      * News Analyst Agent
      * Qwen Chatbot (LangChain + Groq)
      * MedBot — Corrective RAG Medical Assistant
      * Fine-Tune LLAMA 2
      * Voice-Enabled AI Agent — Sun Marke School Assistant
      * Autonomous Data Analyst
    - After listing them, ALWAYS end with: "Which project interests you? I can tell you more about it."
    - When asked about a SPECIFIC project from this list: Provide full details (description, technologies, etc.) found in the context.
    - If asked about a project NOT in this list: Answer based on context or say you don't have that information.

8. **Skills:**
   - Highlight 5-6 most important/impressive skills
   - If specific category asked (e.g., "AI skills", "backend skills"), only mention that category
   - Be concise and impactful
   - No preamble

9. **ABSOLUTELY NO PREAMBLES:**
   - NEVER start with: "I'm here to help", "Let me tell you", "Sure", "Here is", "Here are"
   - Jump STRAIGHT to the answer
   - Examples:
     * BAD: "I'm here to help. What would you like to know about Jamshed's contact details? Email: ..."
     * GOOD: "Email: jamshedalisolangi018@gmail.com..."
     * BAD: "Sure! Here are his skills..."
     * GOOD: "Jamshed specializes in LangChain, LangGraph, RAG, LLMs, and AI Agent Development."

10. **Missing Information:**
    - If info not in context: "I don't have that specific information about Jamshed."
    - **DO NOT** add any extra context, related skills, or "however" statements
    - Just state what's missing and STOP

11. **Off-Topic Questions (jokes, stories, general chat, etc.):**
    - Politely redirect with variety (don't repeat same response)
    - Examples:
      * "I'm here specifically to help with questions about Jamshed's work and experience. What would you like to know about him?"
      * "That's not really my specialty! I focus on answering questions about Jamshed's portfolio. Got any questions about his projects or skills?"
      * "I'm built to talk about Jamshed's expertise, not for that. Anything you'd like to know about his AI work?"

### CRITICAL RULES:
- NEVER use preambles like "I'm here to help", "Sure", "Here is", "Let me tell you"
- Jump DIRECTLY to the answer
- NEVER say "Lower", "But", "Though" when you don't have the answer
- Greetings are the ONLY exception where you can be conversational
- For all other questions: DIRECT ANSWER ONLY

### USER QUESTION:
{question}

Answer:"""

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            ),
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("OK: QA chain setup complete")
        
    def initialize(self):
        """Initialize the complete RAG system with smart caching"""
        print("\n=== Initializing Portfolio RAG Assistant ===\n")
        
        # 1. Setup Pinecone Index
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.setup_pinecone(pc)
        
        # 2. Check if index is already populated
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        if total_vectors > 0:
            print(f"OK: Found {total_vectors} existing vectors in Pinecone. Skipping PDF processing.")
            self.vector_store = PineconeVectorStore(
                index=index,
                embedding=self.embeddings
            )
        else:
            print("💡 Index is empty. Starting PDF processing...")
            chunks = self.load_and_chunk_pdf()
            self.create_vector_store(chunks)
        
        # 3. Setup the QA Chain
        self.setup_qa_chain()
        
        print("\n=== RAG Assistant is ready! ===\n")
        
    def ask(self, question: str, session_id: str = "default_session"):
        """Ask a question about Jamshed Ali with session memory"""
        if not self.qa_chain:
            raise Exception("Please initialize the assistant first using .initialize()")
            
        # Get chat history for this session (list of tuples for ConversationalRetrievalChain)
        history = self.chat_histories.get(session_id, [])
        
        # Invoke the chain
        # NOTE: ConversationalRetrievalChain uses "question" (not "query") and "chat_history"
        response = self.qa_chain.invoke({
            "question": question,
            "chat_history": history
        })
        
        answer = response["answer"]
        
        # Update history (keep last 5 MESSAGES as requested)
        # 1 QA pair = 2 messages. 5 messages = 2.5 pairs.
        # We will keep the last 3 rounds (6 messages) to maintain full context.
        history.append((question, answer))
        if len(history) > 3:
            history = history[-3:]
            
        self.chat_histories[session_id] = history
        
        return answer