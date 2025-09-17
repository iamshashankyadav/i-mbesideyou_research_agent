import requests
from typing import List, Dict
import json

class QuestionGenerator:
    """
    A class to generate questions from research papers using a fine-tuned Hugging Face model.
    """
    
    def __init__(self, model_name: str = "valhalla/t5-small-qg-prepend"):
        """
        Initialize the question generator.
        
        Args:
            model_name (str): Hugging Face model name for question generation
        """
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}  # Replace with actual token if needed
    
    def query_hf_model(self, payload: Dict) -> Dict:
        """
        Query the Hugging Face model API.
        
        Args:
            payload (Dict): Input payload for the model
            
        Returns:
            Dict: Model response
        """
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"Error querying model: {e}")
            return {}
    
    def generate_questions_from_text(self, text: str, num_questions: int = 5) -> List[str]:
        """
        Generate questions from a given text.
        
        Args:
            text (str): Input text to generate questions from
            num_questions (int): Number of questions to generate
            
        Returns:
            List[str]: List of generated questions
        """
        questions = []
        
        # Split text into sentences for better question generation
        sentences = self._split_into_sentences(text)
        
        # Process chunks of text to generate questions
        chunk_size = 3  # Process 3 sentences at a time
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i + chunk_size])
            
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
            
            # Prepare input for question generation model
            input_text = f"generate question: {chunk}"
            
            payload = {
                "inputs": input_text,
                "parameters": {
                    "max_length": 64,
                    "num_return_sequences": 1,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            result = self.query_hf_model(payload)
            
            if result and isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '').strip()
                if generated_text and generated_text not in questions:
                    questions.append(generated_text)
            
            if len(questions) >= num_questions:
                break
        
        return questions[:num_questions]
    
    def generate_questions_by_type(self, text: str, question_types: List[str] = None) -> Dict[str, List[str]]:
        """
        Generate questions by specific types.
        
        Args:
            text (str): Input text
            question_types (List[str]): Types of questions to generate
            
        Returns:
            Dict[str, List[str]]: Questions organized by type
        """
        if question_types is None:
            question_types = ['factual', 'conceptual', 'analytical', 'evaluative']
        
        questions_by_type = {}
        
        for question_type in question_types:
            questions_by_type[question_type] = self._generate_typed_questions(text, question_type)
        
        return questions_by_type
    
    def _generate_typed_questions(self, text: str, question_type: str, num_questions: int = 3) -> List[str]:
        """
        Generate questions of a specific type.
        
        Args:
            text (str): Input text
            question_type (str): Type of question
            num_questions (int): Number of questions to generate
            
        Returns:
            List[str]: Generated questions
        """
        type_prompts = {
            'factual': "Generate factual questions about specific details and facts from:",
            'conceptual': "Generate conceptual questions about theories and concepts from:",
            'analytical': "Generate analytical questions that require analysis from:",
            'evaluative': "Generate evaluative questions that require judgment from:"
        }
        
        prompt = type_prompts.get(question_type, "Generate questions from:")
        input_text = f"{prompt} {text[:800]}"  # Limit text length
        
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_length": 100,
                "num_return_sequences": num_questions,
                "temperature": 0.8,
                "do_sample": True
            }
        }
        
        result = self.query_hf_model(payload)
        questions = []
        
        if result and isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and 'generated_text' in item:
                    question = item['generated_text'].strip()
                    if question and question not in questions:
                        questions.append(question)
        
        return questions
    
    def generate_mcq_questions(self, text: str, num_questions: int = 5) -> List[Dict[str, any]]:
        """
        Generate multiple choice questions.
        
        Args:
            text (str): Input text
            num_questions (int): Number of MCQs to generate
            
        Returns:
            List[Dict]: List of MCQ dictionaries
        """
        mcqs = []
        sentences = self._split_into_sentences(text)
        
        for i in range(min(num_questions, len(sentences) // 3)):
            chunk = " ".join(sentences[i*3:(i+1)*3])
            
            # Generate question
            question_payload = {
                "inputs": f"generate question: {chunk}",
                "parameters": {"max_length": 64, "temperature": 0.7}
            }
            
            question_result = self.query_hf_model(question_payload)
            
            if question_result and isinstance(question_result, list):
                question = question_result[0].get('generated_text', '').strip()
                
                if question:
                    # Generate answer options (simplified approach)
                    options = self._generate_answer_options(chunk, question)
                    
                    mcq = {
                        'question': question,
                        'options': options,
                        'context': chunk[:200] + "..." if len(chunk) > 200 else chunk
                    }
                    mcqs.append(mcq)
        
        return mcqs
    
    def _generate_answer_options(self, context: str, question: str) -> List[str]:
        """
        Generate answer options for MCQ (simplified approach).
        
        Args:
            context (str): Context text
            question (str): Question text
            
        Returns:
            List[str]: Answer options
        """
        # This is a simplified approach - in practice, you'd want a more sophisticated method
        # to generate plausible distractors
        
        # Extract key terms from context
        words = context.split()
        key_terms = [word.strip('.,!?') for word in words if len(word) > 5]
        
        options = []
        if len(key_terms) >= 4:
            options = key_terms[:4]
        else:
            # Fallback options
            options = [
                "Option A",
                "Option B", 
                "Option C",
                "Option D"
            ]
        
        return options
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def generate_study_questions(self, sections: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Generate study questions for different paper sections.
        
        Args:
            sections (Dict[str, str]): Paper sections
            
        Returns:
            Dict[str, List[str]]: Questions organized by section
        """
        study_questions = {}
        
        for section_name, content in sections.items():
            if content.strip():
                questions = self.generate_questions_from_text(content, num_questions=3)
                study_questions[section_name] = questions
        
        return study_questions
    
    def generate_comprehension_questions(self, text: str) -> List[Dict[str, str]]:
        """
        Generate reading comprehension questions.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict[str, str]]: Comprehension questions with difficulty levels
        """
        questions = []
        
        # Generate different difficulty levels
        difficulty_levels = ['easy', 'medium', 'hard']
        
        for difficulty in difficulty_levels:
            if difficulty == 'easy':
                prompt = f"Generate simple factual questions about: {text[:500]}"
            elif difficulty == 'medium':
                prompt = f"Generate questions requiring understanding of concepts from: {text[:500]}"
            else:
                prompt = f"Generate complex analytical questions about: {text[:500]}"
            
            payload = {
                "inputs": prompt,
                "parameters": {"max_length": 80, "temperature": 0.6}
            }
            
            result = self.query_hf_model(payload)
            
            if result and isinstance(result, list):
                for item in result:
                    question_text = item.get('generated_text', '').strip()
                    if question_text:
                        questions.append({
                            'question': question_text,
                            'difficulty': difficulty,
                            'section': 'comprehension'
                        })
        
        return questions