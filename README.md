# Part 3: NLP and Sequence Modeling Mini Project

## Overview

This part builds a complete **NLP pipeline** for customer support sentiment classification. It covers text cleaning, vectorization (TF-IDF and Bag of Words), baseline model training, LSTM architecture design, and a conceptual reflection on attention mechanisms and transformers.

---

## Directory Structure

```
part-3-nlp-sequence-modeling/
│
├── notebook.ipynb                      # Main Jupyter Notebook (all tasks)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
└── results/
    ├── model_evaluation.png            # Confusion matrices (LR + NB)
    ├── model_evaluation.csv            # Classification report (LR)
    └── sample_predictions.txt          # 15 sample predictions with labels
```

---

## Dataset

**File:** `customer_support_text_classification.csv`  
**Records:** 1,500 customer support messages  
**Target:** `sentiment_label` — positive, neutral, negative

| Column | Description |
|---|---|
| `ticket_id` | Unique identifier |
| `customer_message` | Raw text input |
| `sentiment_label` | Target: positive / neutral / negative |
| `channel` | Source channel (chat, phone, email, social) |
| `word_count` | Pre-computed word count |
| `urgent_flag` | Binary urgency flag |

---

## Models

| Model | Vectorization | Accuracy | Weighted F1 |
|---|---|---|---|
| Logistic Regression | TF-IDF (unigrams + bigrams) | ~100% | ~1.00 |
| Naive Bayes | Bag of Words | ~100% | ~1.00 |

> High accuracy reflects that the synthetic dataset has clear keyword signals per class.

---

## Key NLP Concepts

- **Why vectorize?** Models need numbers — text must be converted to numerical form.
- **TF-IDF** weights words by how unique they are to a document vs. the corpus.
- **LSTM** maintains sequential memory via cell state and gated updates.
- **Transformers** use self-attention to capture global context in parallel.
