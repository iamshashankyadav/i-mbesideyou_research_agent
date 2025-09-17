import streamlit as st
import os
from dotenv import load_dotenv
from semantic_scholar import SemanticScholarAPI
from pdf_processor import PDFProcessor
from question_generator import QuestionGenerator
from rag_system import RAGSystem
import json

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Research Paper Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGSystem()

if 'semantic_scholar' not in st.session_state:
    st.session_state.semantic_scholar = SemanticScholarAPI()

if 'pdf_processor' not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()

if 'question_generator' not in st.session_state:
    st.session_state.question_generator = QuestionGenerator()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'uploaded_papers' not in st.session_state:
    st.session_state.uploaded_papers = []

if 'current_paper_content' not in st.session_state:
    st.session_state.current_paper_content = None

if 'paper_sections' not in st.session_state:
    st.session_state.paper_sections = {}

# Main title
st.title("🔬 Research Paper Agent")
st.markdown("*Your AI-powered research assistant for paper discovery, analysis, and interaction*")

# Sidebar
with st.sidebar:
    st.header("📋 Navigation")
    
    # Mode selection
    mode = st.selectbox(
        "Select Mode:",
        ["🔍 Paper Discovery", "📄 Paper Analysis", "💬 Chat with Papers", "❓ Question Generation"]
    )
    
    st.markdown("---")
    
    # API Status
    st.header("🔧 System Status")
    groq_status = "✅ Connected" if os.getenv("GROQ_API_KEY") else "❌ Not configured"
    st.write(f"**GROQ API:** {groq_status}")
    
    # Vector store stats
    if st.session_state.rag_system:
        stats = st.session_state.rag_system.get_stats()
        st.write(f"**Papers in memory:** {len(st.session_state.uploaded_papers)}")
        st.write(f"**Text chunks:** {stats['num_chunks']}")
    
    st.markdown("---")
    
    # Clear data button
    if st.button("🗑️ Clear All Data"):
        st.session_state.rag_system.clear_vector_store()
        st.session_state.chat_history = []
        st.session_state.uploaded_papers = []
        st.session_state.current_paper_content = None
        st.session_state.paper_sections = {}
        st.rerun()

# Main content area
if mode == "🔍 Paper Discovery":
    st.header("🔍 Discover Research Papers")
    st.markdown("Search and explore research papers from Semantic Scholar")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Enter your research query:", placeholder="e.g., machine learning in healthcare")
    
    with col2:
        search_limit = st.number_input("Results", min_value=5, max_value=50, value=10)
    
    # Search filters
    with st.expander("🎛️ Advanced Filters"):
        col1, col2 = st.columns(2)
        with col1:
            min_year = st.number_input("Minimum Year", min_value=1990, max_value=2024, value=2020)
            min_citations = st.number_input("Minimum Citations", min_value=0, value=0)
        with col2:
            max_year = st.number_input("Maximum Year", min_value=1990, max_value=2024, value=2024)
            sort_by = st.selectbox("Sort by", ["Relevance", "Citations", "Year"])
    
    if st.button("🔍 Search Papers", type="primary"):
        if search_query:
            with st.spinner("Searching papers..."):
                papers = st.session_state.semantic_scholar.search_papers(search_query, limit=search_limit)
                
                if papers:
                    # Filter papers
                    filtered_papers = st.session_state.semantic_scholar.filter_papers_by_year(
                        papers, min_year=min_year, max_year=max_year
                    )
                    
                    # Filter by citations
                    if min_citations > 0:
                        filtered_papers = [p for p in filtered_papers if p.get('citationCount', 0) >= min_citations]
                    
                    # Sort papers
                    if sort_by == "Citations":
                        filtered_papers = st.session_state.semantic_scholar.sort_papers_by_citations(filtered_papers)
                    elif sort_by == "Year":
                        filtered_papers = sorted(filtered_papers, key=lambda x: x.get('year', 0), reverse=True)
                    
                    st.success(f"Found {len(filtered_papers)} papers")
                    
                    # Display results
                    for i, paper in enumerate(filtered_papers):
                        with st.expander(f"📄 {paper.get('title', 'Unknown Title')[:100]}..."):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(st.session_state.semantic_scholar.format_paper_info(paper))
                            
                            with col2:
                                if st.button(f"📥 Add to Analysis", key=f"add_{i}"):
                                    st.session_state.uploaded_papers.append({
                                        'title': paper.get('title', 'Unknown'),
                                        'source': 'semantic_scholar',
                                        'data': paper
                                    })
                                    st.success("Paper added to analysis!")
                                
                                if paper.get('url'):
                                    st.markdown(f"[🔗 View Paper]({paper['url']})")
                                
                                # Get recommendations
                                if st.button(f"🎯 Similar Papers", key=f"rec_{i}"):
                                    paper_id = paper.get('paperId')
                                    if paper_id:
                                        recommendations = st.session_state.semantic_scholar.get_recommendations(paper_id)
                                        if recommendations:
                                            st.write("**Similar Papers:**")
                                            for rec in recommendations[:3]:
                                                st.write(f"• {rec.get('title', 'Unknown')}")
                else:
                    st.warning("No papers found for your query. Try different keywords.")
        else:
            st.warning("Please enter a search query.")

elif mode == "📄 Paper Analysis":
    st.header("📄 Paper Analysis")
    st.markdown("Upload and analyze your research papers")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more research papers in PDF format"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file not in [p['file'] for p in st.session_state.uploaded_papers if 'file' in p]:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Extract text
                    text_content = st.session_state.pdf_processor.extract_text_from_pdf(uploaded_file)
                    
                    if text_content:
                        # Clean text
                        cleaned_text = st.session_state.pdf_processor.clean_text(text_content)
                        
                        # Extract sections
                        sections = st.session_state.pdf_processor.extract_sections(cleaned_text)
                        
                        # Get key information
                        key_info = st.session_state.pdf_processor.extract_key_information(cleaned_text)
                        
                        # Add to vector store for RAG
                        st.session_state.rag_system.add_document(
                            cleaned_text,
                            metadata={'filename': uploaded_file.name, 'type': 'uploaded_pdf'}
                        )
                        
                        # Store in session state
                        paper_data = {
                            'filename': uploaded_file.name,
                            'file': uploaded_file,
                            'text_content': cleaned_text,
                            'sections': sections,
                            'key_info': key_info,
                            'processed': True
                        }
                        st.session_state.uploaded_papers.append(paper_data)
                        
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
    
    # Display processed papers
    if st.session_state.uploaded_papers:
        st.subheader("📚 Processed Papers")
        
        for i, paper in enumerate(st.session_state.uploaded_papers):
            with st.expander(f"📄 {paper.get('filename', paper.get('title', f'Paper {i+1}'))}"):
                
                if paper.get('processed'):  # Uploaded PDF
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Paper summary
                        if st.button(f"📝 Generate Summary", key=f"summary_{i}"):
                            with st.spinner("Generating summary..."):
                                summary = st.session_state.pdf_processor.summarize_paper(paper['text_content'])
                                st.markdown("**Summary:**")
                                st.write(summary)
                        
                        # Key information
                        with col1:
                            if paper.get('key_info'):
                                tab1, tab2 = st.tabs(["🔍 Key Information", "📑 Other"])
                                
                                with tab1:
                                    for key, value in paper['key_info'].items():
                                        st.write(f"**{key.title()}:** {value}")

                                with tab2:
                                    st.write("Other content here")

                            

                    with col2:
                        # Actions
                        if st.button(f"💬 Chat about this paper", key=f"chat_{i}"):
                            st.session_state.current_paper_content = paper['text_content']
                            st.session_state.paper_sections = paper['sections']
                            st.info("Paper loaded for chat! Go to 'Chat with Papers' mode.")
                        
                        # Sections
                        if paper.get('sections'):
                            st.write("**Available Sections:**")
                            for section_name, content in paper['sections'].items():
                                if content.strip():
                                    st.write(f"• {section_name.title()}")
                
                else:  # Semantic Scholar paper
                    paper_data = paper.get('data', {})
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Title:** {paper_data.get('title', 'Unknown')}")
                        st.write(f"**Year:** {paper_data.get('year', 'Unknown')}")
                        st.write(f"**Citations:** {paper_data.get('citationCount', 0)}")
                        
                        authors = paper_data.get('authors', [])
                        if authors:
                            author_names = [author.get('name', '') for author in authors]
                            st.write(f"**Authors:** {', '.join(author_names)}")
                        
                        abstract = paper_data.get('abstract', '')
                        if abstract:
                            st.write(f"**Abstract:** {abstract[:300]}...")
                    
                    with col2:
                        if paper_data.get('url'):
                            st.markdown(f"[🔗 View Paper]({paper_data['url']})")

elif mode == "💬 Chat with Papers":
    st.header("💬 Chat with Your Papers")
    st.markdown("Ask questions about your uploaded research papers")
    
    # Check if papers are available
    if not st.session_state.uploaded_papers:
        st.warning("⚠️ No papers available. Please upload papers in the 'Paper Analysis' section first.")
        st.stop()
    
    # Document summary
    with st.expander("📊 Document Summary"):
        summary = st.session_state.rag_system.get_document_summary()
        st.write(summary)
    
    # Suggested questions
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("💡 Suggest Questions"):
            with st.spinner("Generating question suggestions..."):
                suggested_questions = st.session_state.rag_system.suggest_questions()
                if suggested_questions:
                    st.session_state.suggested_questions = suggested_questions
    
    if hasattr(st.session_state, 'suggested_questions'):
        with col2:
            st.write("**Suggested Questions:**")
            for i, question in enumerate(st.session_state.suggested_questions):
                if st.button(f"❓ {question[:50]}...", key=f"suggested_{i}"):
                    st.session_state.current_question = question
    
    # Chat interface
    st.subheader("💬 Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your papers..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare conversation history for context
                conversation_context = [
                    {"query": msg["content"], "response": st.session_state.chat_history[i+1]["content"]}
                    for i, msg in enumerate(st.session_state.chat_history[:-1])
                    if msg["role"] == "user" and i+1 < len(st.session_state.chat_history)
                ]
                
                response = st.session_state.rag_system.chat(prompt, conversation_context)
                st.write(response)
                
                # Add assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

elif mode == "❓ Question Generation":
    st.header("❓ Question Generation")
    st.markdown("Generate study questions from your papers")
    
    if not st.session_state.uploaded_papers:
        st.warning("⚠️ No papers available. Please upload papers in the 'Paper Analysis' section first.")
        st.stop()
    
    # Paper selection
    paper_options = [f"{i+1}. {paper.get('filename', paper.get('title', f'Paper {i+1}'))}" 
                    for i, paper in enumerate(st.session_state.uploaded_papers)]
    
    selected_paper_idx = st.selectbox(
        "Select a paper for question generation:",
        range(len(paper_options)),
        format_func=lambda x: paper_options[x]
    )
    
    if selected_paper_idx is not None:
        selected_paper = st.session_state.uploaded_papers[selected_paper_idx]
        
        # Question generation options
        col1, col2 = st.columns(2)
        
        with col1:
            question_type = st.selectbox(
                "Question Type:",
                ["General Questions", "Multiple Choice", "Section-based", "Comprehension"]
            )
        
        with col2:
            num_questions = st.number_input("Number of Questions:", min_value=3, max_value=15, value=5)
        
        if st.button("🎯 Generate Questions", type="primary"):
            with st.spinner("Generating questions..."):
                
                if question_type == "General Questions":
                    if selected_paper.get('processed'):  # PDF
                        questions = st.session_state.question_generator.generate_questions_from_text(
                            selected_paper['text_content'], num_questions=num_questions
                        )
                    else:  # Semantic Scholar
                        abstract = selected_paper.get('data', {}).get('abstract', '')
                        questions = st.session_state.question_generator.generate_questions_from_text(
                            abstract, num_questions=num_questions
                        )
                    
                    st.subheader("📝 Generated Questions")
                    for i, question in enumerate(questions, 1):
                        st.write(f"**{i}.** {question}")
                
                elif question_type == "Multiple Choice":
                    if selected_paper.get('processed'):
                        mcqs = st.session_state.question_generator.generate_mcq_questions(
                            selected_paper['text_content'], num_questions=num_questions
                        )
                    else:
                        abstract = selected_paper.get('data', {}).get('abstract', '')
                        mcqs = st.session_state.question_generator.generate_mcq_questions(
                            abstract, num_questions=num_questions
                        )
                    
                    st.subheader("🔍 Multiple Choice Questions")
                    for i, mcq in enumerate(mcqs, 1):
                        st.write(f"**Question {i}:** {mcq['question']}")
                        for j, option in enumerate(mcq['options'], 1):
                            st.write(f"   {chr(64+j)}. {option}")
                        
                        with st.expander(f"Context for Question {i}"):
                            st.write(mcq['context'])
                        st.write("---")
                
                elif question_type == "Section-based":
                    if selected_paper.get('sections'):
                        study_questions = st.session_state.question_generator.generate_study_questions(
                            selected_paper['sections']
                        )
                        
                        st.subheader("📚 Section-based Questions")
                        for section, questions in study_questions.items():
                            if questions:
                                st.write(f"**{section.title()} Section:**")
                                for i, question in enumerate(questions, 1):
                                    st.write(f"   {i}. {question}")
                                st.write("")
                    else:
                        st.warning("No sections detected in this paper for section-based questions.")
                
                elif question_type == "Comprehension":
                    if selected_paper.get('processed'):
                        comp_questions = st.session_state.question_generator.generate_comprehension_questions(
                            selected_paper['text_content']
                        )
                    else:
                        abstract = selected_paper.get('data', {}).get('abstract', '')
                        comp_questions = st.session_state.question_generator.generate_comprehension_questions(
                            abstract
                        )
                    
                    st.subheader("🎓 Comprehension Questions")
                    
                    # Group by difficulty
                    difficulty_groups = {}
                    for question in comp_questions:
                        difficulty = question.get('difficulty', 'medium')
                        if difficulty not in difficulty_groups:
                            difficulty_groups[difficulty] = []
                        difficulty_groups[difficulty].append(question['question'])
                    
                    for difficulty in ['easy', 'medium', 'hard']:
                        if difficulty in difficulty_groups:
                            st.write(f"**{difficulty.title()} Level:**")
                            for i, question in enumerate(difficulty_groups[difficulty], 1):
                                st.write(f"   {i}. {question}")
                            st.write("")
        
        # Export questions
        if st.button("📥 Export Questions"):
            # This would generate a downloadable file with all questions
            st.info("Export functionality would generate a downloadable file with all questions.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🔬 Research Paper Agent | Built with Streamlit & AI</p>
        <p><small>Powered by Semantic Scholar API, GROQ, and Hugging Face</small></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Additional styling
st.markdown(
    """
    <style>
    .stExpander > div:first-child {
        background-color: #f0f2f6;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
    }
    
    .stButton > button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)