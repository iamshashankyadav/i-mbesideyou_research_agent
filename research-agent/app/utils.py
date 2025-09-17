"""
Utility functions for the Research Paper Agent
"""

import re
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import streamlit as st

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level (str): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('research_agent.log')
        ]
    )
    return logging.getLogger(__name__)

def clean_filename(filename: str) -> str:
    """
    Clean filename for safe storage.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Cleaned filename
    """
    # Remove special characters
    cleaned = re.sub(r'[^\w\s-.]', '', filename)
    # Replace spaces with underscores
    cleaned = re.sub(r'\s+', '_', cleaned)
    # Limit length
    if len(cleaned) > 100:
        name, ext = cleaned.rsplit('.', 1) if '.' in cleaned else (cleaned, '')
        cleaned = name[:95] + ('.' + ext if ext else '')
    
    return cleaned

def extract_doi(text: str) -> Optional[str]:
    """
    Extract DOI from text.
    
    Args:
        text (str): Text to search for DOI
        
    Returns:
        Optional[str]: DOI if found, None otherwise
    """
    doi_patterns = [
        r'doi:\s*(10\.\d{4,}[^\s]+)',
        r'DOI:\s*(10\.\d{4,}[^\s]+)',
        r'https://doi\.org/(10\.\d{4,}[^\s]+)',
        r'dx\.doi\.org/(10\.\d{4,}[^\s]+)',
    ]
    
    for pattern in doi_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using word overlap.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Similarity score between 0 and 1
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def truncate_text(text: str, max_length: int = 1000, add_ellipsis: bool = True) -> str:
    """
    Truncate text to specified length.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        add_ellipsis (bool): Whether to add ellipsis
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    
    # Try to end at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # If word boundary is not too far back
        truncated = truncated[:last_space]
    
    if add_ellipsis:
        truncated += "..."
    
    return truncated

def format_authors(authors: List[Dict[str, str]]) -> str:
    """
    Format author list for display.
    
    Args:
        authors (List[Dict]): List of author dictionaries
        
    Returns:
        str: Formatted author string
    """
    if not authors:
        return "Unknown Authors"
    
    author_names = [author.get('name', 'Unknown') for author in authors]
    
    if len(author_names) == 1:
        return author_names[0]
    elif len(author_names) == 2:
        return f"{author_names[0]} and {author_names[1]}"
    elif len(author_names) <= 5:
        return ", ".join(author_names[:-1]) + f", and {author_names[-1]}"
    else:
        return ", ".join(author_names[:3]) + f", et al. ({len(author_names)} authors)"

def validate_pdf_file(file) -> Tuple[bool, str]:
    """
    Validate uploaded PDF file.
    
    Args:
        file: Streamlit uploaded file object
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if file is None:
        return False, "No file provided"
    
    # Check file type
    if not file.name.lower().endswith('.pdf'):
        return False, "File must be a PDF"
    
    # Check file size (50MB limit)
    if file.size > 50 * 1024 * 1024:
        return False, "File size must be less than 50MB"
    
    # Basic PDF validation
    try:
        file.seek(0)
        header = file.read(5)
        file.seek(0)
        
        if header != b'%PDF-':
            return False, "Invalid PDF file format"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"
    
    return True, ""

def generate_paper_id(title: str, authors: str = "") -> str:
    """
    Generate unique ID for a paper.
    
    Args:
        title (str): Paper title
        authors (str): Paper authors
        
    Returns:
        str: Unique paper ID
    """
    combined = f"{title.lower().strip()}{authors.lower().strip()}"
    return hashlib.md5(combined.encode()).hexdigest()[:12]

def format_citation_count(count: int) -> str:
    """
    Format citation count for display.
    
    Args:
        count (int): Citation count
        
    Returns:
        str: Formatted citation count
    """
    if count >= 1000:
        return f"{count/1000:.1f}k"
    return str(count)

def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text (str): Input text
        max_keywords (int): Maximum number of keywords
        
    Returns:
        List[str]: Extracted keywords
    """
    # Simple keyword extraction
    import re
    from collections import Counter
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'them', 'their', 'there', 'where', 'when', 'why', 'how', 'what',
        'which', 'who', 'whose', 'whom', 'if', 'then', 'else', 'than', 'as',
        'so', 'too', 'very', 'can', 'may', 'might', 'must', 'shall', 'here'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out stop words and get frequency
    filtered_words = [word for word in words if word not in stop_words]
    word_freq = Counter(filtered_words)
    
    # Get most common words
    return [word for word, _ in word_freq.most_common(max_keywords)]

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string with fallback.
    
    Args:
        json_str (str): JSON string
        default (Any): Default value if parsing fails
        
    Returns:
        Any: Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted file size
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def create_progress_bar(current: int, total: int, description: str = "") -> None:
    """
    Create or update a Streamlit progress bar.
    
    Args:
        current (int): Current progress
        total (int): Total items
        description (str): Progress description
    """
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{description} ({current}/{total})")

def highlight_search_terms(text: str, search_terms: List[str]) -> str:
    """
    Highlight search terms in text for display.
    
    Args:
        text (str): Text to highlight
        search_terms (List[str]): Terms to highlight
        
    Returns:
        str: Text with highlighted terms
    """
    highlighted = text
    
    for term in search_terms:
        if term.lower() in text.lower():
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f"**{term}**", highlighted)
    
    return highlighted

def create_download_link(data: str, filename: str, mime_type: str = "text/plain") -> str:
    """
    Create a download link for data.
    
    Args:
        data (str): Data to download
        filename (str): Filename for download
        mime_type (str): MIME type
        
    Returns:
        str: Download link HTML
    """
    import base64
    
    b64_data = base64.b64encode(data.encode()).decode()
    
    return f'''
    <a href="data:{mime_type};base64,{b64_data}" download="{filename}">
        📥 Download {filename}
    </a>
    '''

def get_timestamp() -> str:
    """
    Get current timestamp string.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class StreamlitSessionManager:
    """Manager for Streamlit session state operations."""
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables."""
        default_values = {
            'rag_system': None,
            'semantic_scholar': None,
            'pdf_processor': None,
            'question_generator': None,
            'chat_history': [],
            'uploaded_papers': [],
            'current_paper_content': None,
            'paper_sections': {},
            'suggested_questions': [],
            'processing_status': {}
        }
        
        for key, default_value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def clear_session_state():
        """Clear all session state."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    
    @staticmethod
    def reset_chat():
        """Reset chat-related session state."""
        st.session_state.chat_history = []
        if hasattr(st.session_state, 'suggested_questions'):
            delattr(st.session_state, 'suggested_questions')
    
    @staticmethod
    def add_paper_to_session(paper_data: Dict[str, Any]):
        """Add paper to session state."""
        if 'uploaded_papers' not in st.session_state:
            st.session_state.uploaded_papers = []
        
        st.session_state.uploaded_papers.append(paper_data)
    
    @staticmethod
    def get_paper_count() -> int:
        """Get number of papers in session."""
        return len(st.session_state.get('uploaded_papers', []))
    
    @staticmethod
    def update_processing_status(paper_id: str, status: str):
        """Update processing status for a paper."""
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {}
        
        st.session_state.processing_status[paper_id] = {
            'status': status,
            'timestamp': get_timestamp()
        }

def display_error_message(message: str, details: str = None):
    """
    Display formatted error message.
    
    Args:
        message (str): Main error message
        details (str): Additional error details
    """
    st.error(f"❌ {message}")
    if details:
        with st.expander("Error Details"):
            st.text(details)

def display_success_message(message: str):
    """
    Display formatted success message.
    
    Args:
        message (str): Success message
    """
    st.success(f"✅ {message}")

def display_info_message(message: str):
    """
    Display formatted info message.
    
    Args:
        message (str): Info message
    """
    st.info(f"ℹ️ {message}")

def display_warning_message(message: str):
    """
    Display formatted warning message.
    
    Args:
        message (str): Warning message
    """
    st.warning(f"⚠️ {message}")