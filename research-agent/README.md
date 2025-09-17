# 📚 Research Agent – AI Tutor with RAG
---

## 🏫 Author & Affiliation
**Name:** Shashank Yadav
**University:** Indian institute of technology Bhilai
**Department:** Data science and artificial intelligence
This repository is part of my internship application project.

---
## 🔖 Overview
This project implements a **Retrieval-Augmented Generation (RAG)**–powered AI Tutor & Research Assistant, designed to help students and researchers interactively explore academic content.
It can:
* **📂 Ingest and process PDF research papers**
* **🔎 Extract key insights and structured sections**
* **💬 Provide chat-based Q&A grounded in documents**
* **📝 Auto-generate summaries and multiple-choice questions**
* **🔗 Fetch external papers from Semantic Scholar API**
* **🎯 Evaluate QA quality metrics for education**
Built using Streamlit, SentenceTransformers, FAISS, and Groq-hosted LLMs (LLaMA-2-70B).
---
## ⚙️ System Architecture
Core components:
* **Frontend:** Streamlit (chat + file upload + visualization)
* **Document Processing:** `pdf_loader.py` + `pdf_processor.py`
* **Vector Storage & Retrieval:** FAISS + `rag_system.py`
* **Generation:** Groq LLM (LLaMA-2-70B)
* **Question Generation:** `question_generator.py`
* **Evaluation:** `qa_evaluation_metrics.py` (structural, relevance, difficulty, accuracy, readability)

---
## 🚀 Installation & Setup
1.  **Clone repo**
    ```
    git clone [https://github.com/](https://github.com/)<your-username>/<repo-name>.git
    cd <repo-name>
    ```
2.  **Create environment**
    ```
    conda create -n ai_tutor python=3.10 -y
    conda activate ai_tutor
    ```
3.  **Install requirements**
    ```
    pip install -r requirements.txt
    ```
4.  **Run Streamlit app**
    ```
    streamlit run app.py
    ```

---
## ✨ Features
* **📄 Upload PDFs:** Cleaned & chunked into a FAISS index.
* **🔍 Chat with documents:** Contextual answers powered by the Groq API.
* **📝 Auto-summary:** Provides a quick overview of research papers.
* **❓ Question Generation:** Creates MCQs with evaluation metrics.
* **📊 QA Evaluation:** Scores based on structural, relevance, difficulty, and accuracy.
* **🌐 Semantic Scholar Integration:** Allows searching for external scientific papers.

---
## 📹 Demo Video/images
🎥 Watch the working demo below:
[Demo](demoandimages/22-01-24.mp4)
For GitHub preview, use a YouTube-like embedded link if you have it hosted externally.
If not, visitors can download & play the `demoandimages` file.
![the interface](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Mississippi_River_at_Lake_Itasca_2016.JPG/800px-Mississippi_River_at_Lake_Itasca_2016.JPG)
---
## 📊 Evaluation Results
The QA generation quality was evaluated using `QAEvaluationMetrics`.
Quick tests gave average scores around 47–49/100 (Poor), indicating the need for dataset improvements and stricter fine-tuning.
**Future improvements:**
* Better curated training data.
* More fine-tuning epochs.
* Post-processing filters for malformed outputs.
## llm interaction logs
https://claude.ai/share/6a6e7183-7ee0-46a0-97c5-e6071ffbcb58
https://chatgpt.com/share/68caefdc-697c-8002-bac9-669aa6569679
---
## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
