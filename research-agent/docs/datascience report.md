### 📊 Data Science Report
---
#### Fine-Tuning Setup
##### **Objective**
The goal was to fine-tune a generative language model to produce high-quality educational QA pairs (questions, multiple-choice options, and answers) grounded in a scientific context.

##### **Data**
* **Source:** A dataset of **2000 examples** loaded from `train_dataset.jsonl`.
* **Format:** The data was formatted with a system prompt for a **scientific question generator** and a user prompt for context. A sample is shown below:
    ```
    <|system|>
    You are a scientific question generator. Create clear, accurate multiple-choice questions from scientific text.

    <|user|>
    Generate a multiple-choice question from the following context.

    Context: Mesophiles grow best in moderate temperature, typically between 25°C and 40°C (77°F and 104°F...
    ```
* **Preprocessing:**
    * The texts were tokenized, with an average length of **201.3 tokens**.

##### **Method**
* **Base Model:** **Phi-3-mini-4k-instruct**.
* **Fine-Tuning Framework:** The pipeline used **QLoRA** for fine-tuning.
* **Training Setup:**
    * **Trainable parameters:** 4,456,448 (0.1165% of total).
    * **Optimizer:** Not explicitly stated, but common for QLoRA is AdamW.
    * **Epochs:** 1.
    * **Batch size:** The log shows 125 steps, implying a batch size of 16 (2000 samples / 125 steps).
* **Training Progress:**
    * The training loss decreased over the 125 steps, from an initial value of 1.7286 to a final value of 0.9300.

##### **Results**
* The **model loaded successfully**, with a trainable percentage of **0.1165%**.
* The training pipeline **completed successfully**, and the adapter was saved to `./qa_adapter` with a size of **17.02 MB**.
* The post-training test **failed** with a `'DynamicCache' object has no attribute 'seen_tokens'` error, indicating a potential issue with the testing script's compatibility with the model's caching mechanism. This error did not affect the training process itself, only the final automated test.
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