# TrustLabel AI - Development Log

## Project Goal

Build an NLP-powered customer support analytics system capable of:

* Intent Classification
* Sentiment Classification
* PII Detection

using Machine Learning and TF-IDF feature engineering.

---

## Phase 1 - Dataset Creation

### Completed

* Created customer support dataset.
* Defined schema:

  * message
  * intent
  * sentiment
  * pii

### Outcome

Initial dataset with 125 labeled customer support messages.

---

## Phase 2 - Exploratory Data Analysis (EDA)

### Completed

* Dataset inspection
* Class distribution analysis
* Missing value analysis
* Data quality validation

### Outcome

Verified dataset consistency and class balance.

---

## Phase 3 - NLP Preprocessing

### Completed

* Lowercase conversion
* Punctuation removal
* Text cleaning
* Feature preparation

### Outcome

Prepared text data for vectorization.

---

## Phase 4 - TF-IDF Feature Engineering

### Completed

* Applied TF-IDF vectorization
* Generated text features for machine learning models

### Outcome

Converted customer messages into numerical feature vectors.

---

## Phase 5 - Intent Classification

### Models Evaluated

* Naive Bayes
* Logistic Regression
* Linear SVM

### Initial Results (125 Records)

| Model               | Accuracy |
| ------------------- | -------: |
| Naive Bayes         |    60.0% |
| Logistic Regression |    52.0% |
| Linear SVM          |    64.0% |

### Key Observation

Model performance was limited by dataset size.

---

## Phase 5A - Dataset Expansion and Optimization

### Completed

* Audited new 500-record dataset
* Compared with original dataset
* Verified schema compatibility
* Verified no duplicate messages
* Merged datasets

### Final Dataset

* Total Records: 625

### Results

| Model               | Accuracy |
| ------------------- | -------: |
| Naive Bayes         |    70.4% |
| Logistic Regression |    72.8% |
| Linear SVM          |    72.0% |

### Key Finding

Increasing dataset size significantly improved model performance.

---

## Phase 6 - Sentiment Classification

### Models Evaluated

* Naive Bayes
* Logistic Regression
* Linear SVM

### Results

| Model               | Accuracy |
| ------------------- | -------: |
| Naive Bayes         |    85.6% |
| Logistic Regression |    88.8% |
| Linear SVM          |    91.2% |

### Best Model

Linear SVM

---

## Phase 7 - PII Detection

### Initial Attempt

Logistic Regression achieved high accuracy but failed to detect PII messages because of class imbalance.

### Improvement

Applied:

```python
class_weight="balanced"
```

### Final Results

* Accuracy: 98.4%
* Recall (PII): 88%
* Precision (PII): 100%

### Key Finding

Class imbalance handling was essential for successful PII detection.

---

## Model Export

### Saved Artifacts

* intent_model.pkl
* intent_vectorizer.pkl
* sentiment_model.pkl
* sentiment_vectorizer.pkl
* pii_model.pkl
* pii_vectorizer.pkl

---

## Current Status

### Completed

* Dataset Creation
* EDA
* NLP Preprocessing
* TF-IDF Feature Engineering
* Intent Classification
* Sentiment Classification
* PII Detection
* Model Export

### Next Steps

* Unified prediction pipeline
* Streamlit UI
* Deployment-ready application

