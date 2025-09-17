### 📊 Data Science Report
---
#### Fine-Tuning Setup
##### **Objective**
The goal was to fine-tune a generative language model to produce high-quality educational QA pairs (questions, multiple-choice options, and answers) grounded in a scientific context.
##### **Data**
* **Source:** Extracted from scientific PDFs and supplementary datasets.
* **Format:** Each sample contained:
    * **Context:** A paragraph or section from a scientific paper.
    * **Question:** A generated or labeled question.
    * **Options:** Multiple-choice answers (when available).
    * **Answer:** Correct answer(s).
* **Preprocessing:**
    * Text cleaning (punctuation fixes, sentence splitting).
    * Context chunking (~200–300 tokens).
    * Filtering out very short or incoherent samples.
##### **Method**
* **Base Model:** Phi-3-mini-4k-instruct (chosen for efficiency and instruction-following ability).
* **Fine-Tuning Framework:** Hugging Face transformers + **PEFT** (Parameter Efficient Fine-Tuning).
* **Training Setup:**
    * **Optimizer:** AdamW
    * **Learning rate:** 5e-5
    * **Epochs:** 3–5
    * **Batch size:** 16
    * **LoRA** adapters to reduce memory footprint.
##### **Results**
* The model loaded successfully after fine-tuning.
* Early qualitative inspection shows it can generate structured questions with options, but question quality varies (some are incomplete, truncated, or incoherent).
* Quantitative evaluation was handled with **QAEvaluationMetrics** (see Section 2).
---
#### Evaluation Methodology & Outcomes
##### **Methodology**
We used a custom **QAEvaluationMetrics** framework, which scores generated questions across:
* **Structure:** Presence of ?, options, and a correct answer label.
* **Relevance:** Overlap with the provided context.
* **Difficulty:** Bloom’s taxonomy.
* **Scientific Accuracy:** Terminology consistency.
* **Readability:** Sentence length heuristic.
* **Overall Score:** A weighted composite.
Each generated question receives individual metric scores and a quality rating (**Excellent** / **Good** / **Fair** / **Poor**).
##### **Quantitative Outcomes**
| Test | Question Example | Overall Score | Quality Rating |
| :--- | :--- | :--- | :--- |
| 1 | "does photosynthesis produce?" | 49.2/100 | Poor |
| 2 | "oth strands serve in the formation of?" | 49.4/100 | Poor |
| 3 | "n do you have to take if your body is infected with an uncontrollable virus?" | 47.0/100 | Poor |
##### **Aggregate Findings**
* **Average score:** ~48/100 (in the **Poor** range).
* **Common issues:**
    * Missing or malformed options.
    * Lack of an explicit correct answer indication.
    * Truncated or incomplete phrasing.
    * Weak alignment with the provided scientific context.
##### **Qualitative Outcomes**
* **Strengths:**
    * The model produces questions that resemble multiple-choice formats.
    * It captures some domain-specific terms (e.g., “photosynthesis”).
* **Weaknesses:**
    * Incomplete grammar and phrasing.
    * Frequent absence of the “Answer:” marker.
    * Inconsistent scientific accuracy.
---
#### Next Steps
* **Improve the training dataset:**
    * Ensure well-formed questions with explicit answers.
    * Include negative examples (badly formed QAs) for contrastive learning.
    * Expand domain coverage.
* Experiment with longer fine-tuning (more epochs, larger context windows).
* Add post-processing filters to automatically discard malformed generations.
* Evaluate with human experts for qualitative validation beyond automated metrics.