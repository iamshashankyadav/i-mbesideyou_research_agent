import fitz  # PyMuPDF
from typing import List, Dict
import re
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PDFProcessor:
    """
    A class to process PDF files and extract content for research paper analysis.
    """

    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text content from a PDF file using PyMuPDF (fitz).
        Works with uploaded file objects or file paths.
        """
        try:
            # Handle both file path and file-like objects
            if isinstance(pdf_file, str):
                doc = fitz.open(pdf_file)
            else:
                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            return text.strip()

        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 3 and not line.isdigit():
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def extract_sections(self, text: str) -> Dict[str, str]:
        sections = {}
        section_patterns = {
            'abstract': r'(?i)abstract\s*\n(.*?)(?=\n(?:introduction|keywords|\d+\.?\s*introduction))',
            'introduction': r'(?i)(?:\d+\.?\s*)?introduction\s*\n(.*?)(?=\n(?:\d+\.?\s*\w+|references|bibliography))',
            'methodology': r'(?i)(?:\d+\.?\s*)?(?:methodology|methods|approach)\s*\n(.*?)(?=\n(?:\d+\.?\s*\w+|references|bibliography))',
            'results': r'(?i)(?:\d+\.?\s*)?(?:results|findings)\s*\n(.*?)(?=\n(?:\d+\.?\s*\w+|references|bibliography))',
            'conclusion': r'(?i)(?:\d+\.?\s*)?(?:conclusion|conclusions)\s*\n(.*?)(?=\n(?:references|bibliography))',
            'references': r'(?i)(?:references|bibliography)\s*\n(.*?)$'
        }
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            sections[section_name] = match.group(1).strip() if match else ""
        return sections

    def summarize_paper(self, text: str) -> str:
        """
        Generate a summary of the research paper using Groq API.
        
        Args:
            text (str): Full paper text
            
        Returns:
            str: Generated summary
        """
        try:
            # Truncate text if too long (Groq has token limits)
            max_chars = 8000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            prompt = f"""
            Please provide a comprehensive summary of this research paper. Include:
            1. Main research question/objective
            2. Methodology used
            3. Key findings
            4. Conclusions and implications
            5. Limitations (if mentioned)
            
            Paper content:
            {text}
            
            Summary:
            """
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert research assistant specializing in academic paper analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Unable to generate summary. Please try again."
    
    def extract_key_information(self, text: str) -> Dict[str, str]:
        """
        Extract key information from the paper.
        
        Args:
            text (str): Full paper text
            
        Returns:
            Dict[str, str]: Key information extracted
        """
        try:
            prompt = f"""
            Extract the following key information from this research paper:
            1. Title (if identifiable)
            2. Research domain/field
            3. Main keywords (5-10 keywords)
            4. Research methodology
            5. Dataset used (if any)
            6. Key contributions
            
            Format the response as a structured text with clear labels.
            
            Paper content (first 3000 characters):
            {text[:3000]}
            """
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting key information from academic papers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.2
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the structured response
            info = {}
            lines = result.split('\n')
            current_key = ""
            
            for line in lines:
                if ':' in line and len(line.split(':')) == 2:
                    key, value = line.split(':', 1)
                    current_key = key.strip().lower()
                    info[current_key] = value.strip()
                elif current_key and line.strip():
                    info[current_key] += " " + line.strip()
            
            return info
            
        except Exception as e:
            print(f"Error extracting key information: {e}")
            return {}
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split text into chunks for RAG processing.
        
        Args:
            text (str): Full text to chunk
            chunk_size (int): Size of each chunk
            overlap (int): Overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Find the last complete sentence in the chunk
            last_sentence = max(
                chunk.rfind('.'),
                chunk.rfind('!'),
                chunk.rfind('?')
            )
            
            if last_sentence != -1 and last_sentence > len(chunk) * 0.5:
                chunk = chunk[:last_sentence + 1]
                end = start + last_sentence + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def get_paper_metadata(self, text: str) -> Dict[str, str]:
        """
        Extract metadata from the paper.
        
        Args:
            text (str): Paper text
            
        Returns:
            Dict[str, str]: Metadata information
        """
        metadata = {}
        
        # Extract potential title (first few lines)
        lines = text.split('\n')[:10]
        for line in lines:
            if len(line.strip()) > 10 and not line.strip().isdigit():
                metadata['potential_title'] = line.strip()
                break
        
        # Count pages (rough estimate)
        page_breaks = text.count('\x0c') or text.count('Page ')
        metadata['estimated_pages'] = str(max(1, page_breaks))
        
        # Estimate word count
        words = len(text.split())
        metadata['word_count'] = str(words)
        
        # Extract potential DOI
        doi_pattern = r'(?i)doi\s*:?\s*(10\.\d{4,}[^\s]+)'
        doi_match = re.search(doi_pattern, text)
        if doi_match:
            metadata['doi'] = doi_match.group(1)
        
        return metadata