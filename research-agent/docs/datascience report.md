📊 Data Science Report
1. Fine-Tuning Setup
Objective

The goal was to fine-tune a generative language model to produce high-quality educational QA pairs (questions, multiple-choice options, and answers) grounded in scientific context.

Data

Source: Extracted from scientific PDFs and supplementary datasets.

Format:

Each sample contained:

Context: Paragraph or section from scientific paper.

Question: Generated or labeled question.

Options: Multiple-choice answers (when available).

Answer: Correct answer(s).

Preprocessing:

Text cleaning (punctuation fixes, sentence splitting).

Context chunking (~200–300 tokens).

Filtering out very short or incoherent samples.

Method

Base Model: Phi-3-mini-4k-instruct (chosen for efficiency and instruction-following ability).

Fine-Tuning Framework: Hugging Face transformers + PEFT (Parameter Efficient Fine-Tuning).

Training Setup:

Optimizer: AdamW

Learning rate: 5e-5

Epochs: 3–5

Batch size: 16

LoRA adapters to reduce memory footprint.

Results

Model loaded successfully after fine-tuning.

Early qualitative inspection shows it can generate structured questions with options, but question quality varies (some incomplete, truncated, or incoherent).

Quantitative evaluation handled with QAEvaluationMetrics (see Section 2).

2. Evaluation Methodology & Outcomes
Methodology

We used the custom QAEvaluationMetrics framework, which scores generated questions across:

Structure (presence of ?, options, correct answer label).

Relevance (overlap with context).

Difficulty (Bloom’s taxonomy).

Scientific Accuracy (terminology consistency).

Readability (sentence length heuristic).

Overall Score (weighted composite).

Each generated question receives:

Individual metric scores.

Quality rating: Excellent / Good / Fair / Poor.

Quantitative Outcomes

📝 Quick Test Results:

Test	Question Example	Overall Score	Quality Rating
1	"does photosynthesis produce?"	49.2/100	Poor
2	"oth strands serve in the formation of?"	49.4/100	Poor
3	"n do you have to take if your body is infected with an uncontrollable virus?"	47.0/100	Poor

Aggregate Findings:

Average score: ~48/100 (Poor range).

Common issues:

Missing or malformed options.

Lack of explicit correct answer indication.

Truncated/incomplete phrasing.

Weak alignment with provided scientific context.

Qualitative Outcomes

Strengths:

Model produces questions that resemble multiple-choice formats.

Captures some domain-specific terms (e.g., “photosynthesis”).

Weaknesses:

Incomplete grammar/phrasing.

Frequent absence of “Answer:” marker.

Inconsistent scientific accuracy.

3. Next Steps

Improve training dataset by:

Ensuring well-formed questions with explicit answers.

Including negative examples (badly formed QAs) for contrastive learning.

Expanding domain coverage.

Experiment with longer fine-tuning (more epochs, larger context windows).

Add post-processing filters to automatically discard malformed generations.

Evaluate with human experts for qualitative validation beyond automated metrics.