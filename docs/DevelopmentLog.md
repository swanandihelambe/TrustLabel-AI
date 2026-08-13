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

* Applied TF-IDF vectorization.
* Generated text features for machine learning models.

### Outcome

Converted customer messages into numerical feature vectors.

---

## Phase 5 - Intent Classification

### Models Evaluated

* Naive Bayes
* Logistic Regression
* Linear SVM

### Initial Results (125 Records)

| Model | Accuracy |
|---|---:|
| Naive Bayes | 60.0% |
| Logistic Regression | 52.0% |
| Linear SVM | 64.0% |

### Key Observation

Model performance was limited by dataset size.

---

## Phase 5A - Dataset Expansion and Optimization

### Completed

* Audited new 500-record dataset.
* Compared with original dataset.
* Verified schema compatibility.
* Verified no duplicate messages.
* Merged datasets.

### Intermediate Dataset

* Total Records: 625
* Intent Classes: 5

### Results

| Model | Accuracy |
|---|---:|
| Naive Bayes | 70.4% |
| Logistic Regression | 72.8% |
| Linear SVM | 72.0% |

### Key Finding

Increasing dataset size significantly improved model performance.

---

## Phase 5B - Out-of-Scope Classification

### Completed

An additional intent class, **Out-of-Scope**, was introduced to prevent the classifier from forcing unrelated messages into one of the customer-support categories.

### Final Dataset

* Total Records: 750
* Total Intent Classes: 6

### Intent Distribution

* Account: 125
* Billing: 125
* Security: 125
* Subscription: 125
* Technical: 125
* Out-of-Scope: 125

### Final Model

**Logistic Regression**

### Final Results

**Accuracy: 80.7%**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Account | 0.74 | 0.68 | 0.71 |
| Billing | 0.83 | 0.60 | 0.70 |
| Out-of-Scope | 0.85 | 0.88 | 0.86 |
| Security | 0.79 | 0.76 | 0.78 |
| Subscription | 0.96 | 1.00 | 0.98 |
| Technical | 0.70 | 0.92 | 0.79 |

### Key Finding

Adding the Out-of-Scope class improved the intent classification result from the previous 72.8% accuracy to 80.7% and allowed the system to explicitly identify messages outside the supported customer-support domain.

---

## Phase 6 - Sentiment Classification

### Models Evaluated

* Naive Bayes
* Logistic Regression
* Linear SVM

### Results

| Model | Accuracy |
|---|---:|
| Naive Bayes | 85.6% |
| Logistic Regression | 88.8% |
| Linear SVM | 91.2% |

### Best Model

**LinearSVC**

### Final Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Negative | 0.94 | 0.92 | 0.93 |
| Neutral | 0.88 | 0.95 | 0.91 |
| Positive | 0.93 | 0.84 | 0.88 |

---

## Phase 7 - PII Detection

### Initial Observation

The initial PII model showed that accuracy alone was not sufficient because the dataset contained substantially more non-PII examples.

### Final Model

**Logistic Regression**

### Final Results

* Accuracy: 96.8%
* PII Precision: 100%
* PII Recall: 75%
* PII F1-Score: 86%

### Key Finding

PII recall is particularly important because false negatives can allow sensitive information to pass undetected. Therefore, accuracy alone is not used as the sole indicator of PII model performance.

---

## Phase 8 - Model Export

### Saved Artifacts

* intent_model.pkl
* intent_vectorizer.pkl
* sentiment_model.pkl
* sentiment_vectorizer.pkl
* pii_model.pkl
* pii_vectorizer.pkl

### Verification

Model-vectorizer feature compatibility was verified for all three classification pipelines.

---

## Phase 9 - Unified Prediction Pipeline

### Completed

Implemented a unified prediction pipeline that accepts a customer support message and generates:

* Intent
* Sentiment
* PII Detection Result

The pipeline also supports the Out-of-Scope intent.

### Prediction Testing

Tested messages covering:

* Account-related requests
* Billing-related requests
* Technical issues
* Messages containing PII
* Out-of-Scope messages

The prediction pipeline produced the expected classifications for the tested examples.

---

## Phase 10 - Streamlit Application

### Completed

Developed a Streamlit interface providing:

* Common customer-support message examples
* Custom message input
* One-click analysis
* Intent prediction
* Sentiment prediction
* PII detection
* Out-of-Scope handling
* Model performance information
* Detailed evaluation metrics
* Architecture overview
* Project limitations

---

## Final Project Status

### Completed

* Dataset Creation
* EDA
* NLP Preprocessing
* TF-IDF Feature Engineering
* Intent Classification
* Out-of-Scope Classification
* Sentiment Classification
* PII Detection
* Model Export
* Unified Prediction Pipeline
* Streamlit Application

### Final Results

| Component | Final Result |
|---|---:|
| Dataset | 750 records |
| Intent Classes | 6 |
| Intent Accuracy | 80.7% |
| Sentiment Accuracy | 91.2% |
| PII Accuracy | 96.8% |

### Current Status

**Core development completed and application ready for portfolio presentation.**
