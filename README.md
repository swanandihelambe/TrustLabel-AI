# TrustLabel AI

An NLP-powered customer support analytics system that automatically classifies customer messages by intent, sentiment, and PII (Personally Identifiable Information).

The project was inspired by a Data Labeling Analyst job simulation and focuses on applying Natural Language Processing and Machine Learning techniques to real-world customer support workflows.

---

## Features

### Intent Classification

Classifies customer messages into:

* Billing
* Account
* Technical
* Security
* Subscription

### Sentiment Analysis

Detects customer sentiment:

* Positive
* Neutral
* Negative

### PII Detection

Identifies messages containing personally identifiable information such as:

* Email addresses
* Phone numbers
* Sensitive account details

---

## Dataset

### Original Dataset

* 125 labeled customer support messages

### Expanded Dataset

* 500 additional labeled messages

### Final Master Dataset

* 625 customer support messages
* Balanced intent distribution
* Intent, Sentiment, and PII labels

Schema:

```text
message
intent
sentiment
pii
```

---

## Machine Learning Pipeline

### NLP Preprocessing

* Lowercase conversion
* Text cleaning
* Punctuation removal
* Feature preparation

### Feature Engineering

* TF-IDF Vectorization

### Models Evaluated

* Multinomial Naive Bayes
* Logistic Regression
* Linear SVM

---

## Results

### Intent Classification

| Model               |  Accuracy |
| ------------------- | --------: |
| Naive Bayes         |     70.4% |
| Logistic Regression | **72.8%** |
| Linear SVM          |     72.0% |

Best Model: **Logistic Regression**

---

### Sentiment Classification

| Model               |  Accuracy |
| ------------------- | --------: |
| Naive Bayes         |     85.6% |
| Logistic Regression |     88.8% |
| Linear SVM          | **91.2%** |

Best Model: **Linear SVM**

---

### PII Detection

| Model                        |  Accuracy |
| ---------------------------- | --------: |
| Logistic Regression          |     87.2% |
| Balanced Logistic Regression | **98.4%** |

Additional PII Metrics:

* Precision: 100%
* Recall: 88%

Best Model: **Balanced Logistic Regression**

---

## Key Findings

* Increasing dataset size from 125 to 625 records significantly improved model performance.
* Logistic Regression achieved the best intent classification performance.
* Linear SVM achieved the highest sentiment classification accuracy.
* TF-IDF feature expansion with bigrams did not improve performance.
* Accuracy alone was misleading for imbalanced PII datasets.
* Class balancing dramatically improved PII detection quality.

---

## Project Structure

```text
trustlabel-ai/
│
├── data/
├── docs/
├── images/
├── models/
├── notebooks/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Saved Model Artifacts

```text
intent_model.pkl
intent_vectorizer.pkl

sentiment_model.pkl
sentiment_vectorizer.pkl

pii_model.pkl
pii_vectorizer.pkl
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Jupyter Notebook

---

## Future Enhancements

- Unified prediction pipeline for Intent, Sentiment, and PII analysis
- Streamlit-based interactive web application
- Prediction confidence scores
- Expanded customer support dataset for improved model performance

---
