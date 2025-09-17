import os
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from dotenv import load_dotenv
import pickle
import json

# Load environment variables
load_dotenv()

class RAGSystem:
    """
    Retrieval-Augmented Generation system for research paper chatbot functionality.
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model_name (str): Name of the sentence transformer model
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Vector store components
        self.index = None
        self.texts = []
        self.metadata = []
        self.embeddings = None
        
        # Configuration
        self.chunk_size = 1000
        self.chunk_overlap = 100
        self.top_k = 5
    
    def add_document(self, text: str, metadata: Dict = None):
        """
        Add a document to the vector store.
        
        Args:
            text (str): Document text
            metadata (Dict): Document metadata
        """
        if metadata is None:
            metadata = {}
        
        # Split document into chunks
        chunks = self._chunk_text(text)
        
        # Generate embeddings for chunks
        chunk_embeddings = self.embedding_model.encode(chunks)
        
        # Add to existing data
        self.texts.extend(chunks)
        self.metadata.extend([metadata for _ in chunks])
        
        # Update embeddings
        if self.embeddings is None:
            self.embeddings = chunk_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, chunk_embeddings])
        
        # Rebuild FAISS index
        self._build_index()
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: Text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
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
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def _build_index(self):
        """Build or rebuild the FAISS index."""
        if self.embeddings is not None and len(self.embeddings) > 0:
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float, Dict]]:
        """
        Search for relevant chunks based on query.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[Tuple[str, float, Dict]]: List of (text, score, metadata) tuples
        """
        if top_k is None:
            top_k = self.top_k
        
        if self.index is None or len(self.texts) == 0:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append((
                    self.texts[idx],
                    float(score),
                    self.metadata[idx]
                ))
        
        return results
    
    def generate_response(self, query: str, context_chunks: List[str] = None) -> str:
        """
        Generate a response using retrieved context.
        
        Args:
            query (str): User query
            context_chunks (List[str]): Retrieved context chunks
            
        Returns:
            str: Generated response
        """
        if context_chunks is None:
            # Retrieve relevant chunks
            search_results = self.search(query)
            context_chunks = [result[0] for result in search_results]
        
        # Prepare context
        context = "\n\n".join(context_chunks[:3])  # Use top 3 chunks
        
        # Create prompt
        prompt = f"""
        Based on the following context from research papers, please answer the question accurately and comprehensively.
        
        Context:
        {context}
        
        Question: {query}
        
        Instructions:
        - Provide a detailed answer based on the context
        - If the context doesn't contain enough information, say so
        - Include relevant details and explanations
        - Be precise and academic in tone
        
        Answer:
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert research assistant who provides accurate, detailed answers based on academic content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm unable to generate a response at this time. Please try again."
    
    def chat(self, query: str, conversation_history: List[Dict] = None) -> str:
        """
        Main chat interface for the RAG system.
        
        Args:
            query (str): User query
            conversation_history (List[Dict]): Previous conversation messages
            
        Returns:
            str: Response
        """
        # Search for relevant context
        search_results = self.search(query, top_k=5)
        
        if not search_results:
            return "I don't have enough information in the uploaded documents to answer your question. Please make sure you've uploaded relevant research papers."
        
        # Extract context and prepare response
        context_chunks = [result[0] for result in search_results[:3]]  # Filter by relevance score
        
        if not context_chunks:
            return "I couldn't find sufficiently relevant information in the documents to answer your question. Could you rephrase or provide more specific details?"
        
        # Include conversation history in prompt if available
        history_context = ""
        if conversation_history:
            recent_history = conversation_history[-3:]  # Last 3 exchanges
            history_context = "\n".join([
                f"Previous Q: {msg['query']}\nPrevious A: {msg['response']}"
                for msg in recent_history if 'query' in msg and 'response' in msg
            ])
        
        context = "\n\n".join(context_chunks[:3])
        
        prompt = f"""
        You are an expert research assistant helping with questions about uploaded research papers.
        
        {"Previous conversation context:" + history_context if history_context else ""}
        
        Relevant document context:
        {context}
        
        Current question: {query}
        
        Instructions:
        - Answer based primarily on the provided context
        - Be precise and academic in tone
        - If you need clarification, ask specific follow-up questions
        - If the context is insufficient, clearly state what additional information would be needed
        - Cite specific parts of the context when relevant
        
        Response:
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert research assistant specializing in academic paper analysis and discussion."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.4
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return "I encountered an error while processing your question. Please try again."
    
    def get_document_summary(self) -> str:
        """
        Get a summary of all documents in the vector store.
        
        Returns:
            str: Summary of documents
        """
        if not self.texts:
            return "No documents have been added to the system."
        
        # Sample some chunks for summary
        sample_texts = self.texts[:min(5, len(self.texts))]
        combined_text = "\n\n".join(sample_texts)
        
        prompt = f"""
        Provide a brief overview of the research documents based on these sample excerpts:
        
        {combined_text}
        
        Include:
        - Main research areas/topics
        - Key themes
        - Document types (if identifiable)
        
        Summary:
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert at summarizing academic content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating document summary: {e}")
            return "Unable to generate document summary."
    
    def suggest_questions(self, num_questions: int = 5) -> List[str]:
        """
        Suggest questions based on the document content.
        
        Args:
            num_questions (int): Number of questions to suggest
            
        Returns:
            List[str]: Suggested questions
        """
        if not self.texts:
            return []
        
        # Use a sample of texts to generate questions
        sample_size = min(3, len(self.texts))
        sample_texts = self.texts[:sample_size]
        combined_text = " ".join(sample_texts)[:2000]  # Limit length
        
        prompt = f"""
        Based on this research content, suggest {num_questions} interesting questions that someone might ask:
        
        Content:
        {combined_text}
        
        Generate questions that are:
        - Specific to the content
        - Thought-provoking
        - Answerable from the material
        - Varied in complexity
        
        Questions (one per line):
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert at generating insightful questions about academic content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.6
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip() and not q.strip().startswith(('1.', '2.', '3.', '4.', '5.'))]
            
            # Clean up questions (remove numbering)
            cleaned_questions = []
            for q in questions:
                # Remove leading numbers and special characters
                cleaned = q.lstrip('0123456789.- ').strip()
                if cleaned and len(cleaned) > 10:
                    cleaned_questions.append(cleaned)
            
            return cleaned_questions[:num_questions]
            
        except Exception as e:
            print(f"Error suggesting questions: {e}")
            return []
    
    def save_vector_store(self, filepath: str):
        """
        Save the vector store to disk.
        
        Args:
            filepath (str): Path to save the vector store
        """
        data = {
            'texts': self.texts,
            'metadata': self.metadata,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Vector store saved to {filepath}")
    
    def load_vector_store(self, filepath: str):
        """
        Load the vector store from disk.
        
        Args:
            filepath (str): Path to load the vector store from
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.texts = data['texts']
            self.metadata = data['metadata']
            
            if data['embeddings'] is not None:
                self.embeddings = np.array(data['embeddings'])
                self._build_index()
            
            print(f"Vector store loaded from {filepath}")
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
    
    def clear_vector_store(self):
        """Clear all data from the vector store."""
        self.texts = []
        self.metadata = []
        self.embeddings = None
        self.index = None
        print("Vector store cleared.")
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict[str, any]: Statistics
        """
        return {
            'num_chunks': len(self.texts),
            'total_characters': sum(len(text) for text in self.texts),
            'average_chunk_length': np.mean([len(text) for text in self.texts]) if self.texts else 0,
            'has_index': self.index is not None,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0
        }