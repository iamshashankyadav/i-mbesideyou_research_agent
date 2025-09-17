"""
Configuration settings for the Research Paper Agent
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Research Paper Agent"""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Application Settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Model Configuration
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    QUESTION_MODEL = os.getenv("QUESTION_MODEL", "valhalla/t5-small-qg-prepend")
    GROQ_MODEL = "llama3-70b-8192"
    
    # Vector Store Settings
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store.pkl")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # Search and Retrieval Settings
    DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", "10"))
    RAG_TOP_K = 5
    SIMILARITY_THRESHOLD = 0.3
    
    # Chat Settings
    MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", "10"))
    MAX_TOKENS_SUMMARY = 1000
    MAX_TOKENS_CHAT = 2000
    MAX_TOKENS_QUESTIONS = 800
    
    # Semantic Scholar API Settings
    SS_BASE_URL = "https://api.semanticscholar.org/graph/v1"
    SS_RATE_LIMIT = 100  # requests per 5 minutes
    SS_DEFAULT_FIELDS = [
        'paperId', 'title', 'abstract', 'authors', 'year', 'citationCount',
        'influentialCitationCount', 'venue', 'url', 'publicationDate'
    ]
    
    # PDF Processing Settings
    MAX_PDF_SIZE_MB = 50
    SUPPORTED_FORMATS = ['pdf']
    
    # Question Generation Settings
    DEFAULT_NUM_QUESTIONS = 5
    MAX_QUESTIONS = 20
    QUESTION_TYPES = ['factual', 'conceptual', 'analytical', 'evaluative']
    
    # UI Settings
    PAGE_TITLE = "Research Paper Agent"
    PAGE_ICON = "📚"
    LAYOUT = "wide"
    
    @classmethod
    def validate_config(cls):
        """Validate essential configuration settings"""
        errors = []
        
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is not set in environment variables")
        
        if cls.CHUNK_SIZE < 100:
            errors.append("CHUNK_SIZE should be at least 100 characters")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP should be less than CHUNK_SIZE")
        
        if cls.MAX_CHAT_HISTORY < 1:
            errors.append("MAX_CHAT_HISTORY should be at least 1")
        
        if errors:
            raise ValueError("Configuration errors found:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True
    
    @classmethod
    def get_streamlit_config(cls):
        """Get Streamlit-specific configuration"""
        return {
            "page_title": cls.PAGE_TITLE,
            "page_icon": cls.PAGE_ICON,
            "layout": cls.LAYOUT,
            "initial_sidebar_state": "expanded"
        }
    
    @classmethod
    def get_groq_config(cls):
        """Get GROQ-specific configuration"""
        return {
            "api_key": cls.GROQ_API_KEY,
            "model": cls.GROQ_MODEL,
            "max_tokens_summary": cls.MAX_TOKENS_SUMMARY,
            "max_tokens_chat": cls.MAX_TOKENS_CHAT,
            "max_tokens_questions": cls.MAX_TOKENS_QUESTIONS
        }
    
    @classmethod
    def get_rag_config(cls):
        """Get RAG system configuration"""
        return {
            "embedding_model": cls.EMBEDDINGS_MODEL,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "top_k": cls.RAG_TOP_K,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD
        }
    
    @classmethod
    def get_semantic_scholar_config(cls):
        """Get Semantic Scholar API configuration"""
        return {
            "base_url": cls.SS_BASE_URL,
            "rate_limit": cls.SS_RATE_LIMIT,
            "default_fields": cls.SS_DEFAULT_FIELDS,
            "default_limit": cls.DEFAULT_SEARCH_LIMIT
        }

# Validate configuration on import
try:
    Config.validate_config()
except ValueError as e:
    print(f"Configuration Warning: {e}")