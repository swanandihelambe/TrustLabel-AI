# TrustLabel AI

**TrustLabel AI** is an NLP-based customer support analytics system that analyzes customer messages across three dimensions:

* **Intent Classification** — identifies the type of customer request.
* **Sentiment Classification** — determines whether the message is Positive, Neutral, or Negative.
* **PII Detection** — identifies whether personally identifiable information is present.

The system uses **TF-IDF feature engineering and supervised Machine Learning models** to automate customer support message analysis.

---

## Demo

TrustLabel AI provides an interactive **Streamlit web application** where users can:

* Select common customer-support messages
* Enter custom customer messages
* Analyze messages with one click
* View predicted intent, sentiment, and PII status
* Identify messages outside the supported customer-support domain
* Review model evaluation metrics

---

## Key Results

| Task | Selected Model | Accuracy |
|---|---|---:|
| Intent Classification | Logistic Regression | **80.7%** |
| Sentiment Classification | LinearSVC | **91.2%** |
| PII Detection | Logistic Regression | **96.8%** |

**Final Dataset:** 750 labeled customer support messages

**Intent Classes:** 6

### Intent Classes

* Account
* Billing
* Security
* Subscription
* Technical
* Out-of-Scope

The Out-of-Scope class allows the system to identify messages that do not belong to the supported customer-support domain.

---

## Model Evaluation

### Intent Classification

**Logistic Regression — 80.7% Accuracy**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Account | 0.74 | 0.68 | 0.71 |
| Billing | 0.83 | 0.60 | 0.70 |
| Out-of-Scope | 0.85 | 0.88 | 0.86 |
| Security | 0.79 | 0.76 | 0.78 |
| Subscription | 0.96 | 1.00 | 0.98 |
| Technical | 0.70 | 0.92 | 0.79 |

### Sentiment Classification

**LinearSVC — 91.2% Accuracy**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Negative | 0.94 | 0.92 | 0.93 |
| Neutral | 0.88 | 0.95 | 0.91 |
| Positive | 0.93 | 0.84 | 0.88 |

### PII Detection

**Logistic Regression — 96.8% Accuracy**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| No PII | 0.96 | 1.00 | 0.98 |
| PII Present | 1.00 | 0.75 | 0.86 |

> **Important:** PII accuracy alone does not tell the full story because the test set contains substantially more non-PII examples. The model achieved **75% recall on PII-positive samples**, so it should not be treated as a security guarantee or used as the sole mechanism for protecting sensitive information.

---

## Features

* Intent Classification
* Out-of-Scope Detection
* Sentiment Analysis
* PII Detection
* Text Preprocessing
* TF-IDF Feature Engineering
* Multi-model Evaluation
* Model and Vectorizer Export
* Unified Prediction Pipeline
* Interactive Streamlit Application

---

## Machine Learning Pipeline

```text
Customer Support Message
            |
            v
    Text Preprocessing
            |
            v
         TF-IDF
            |
     +------+------+
     |      |      |
     v      v      v
  Intent Sentiment PII
   Model    Model   Model
     |      |      |
     +------+------+
            |
            v
     Unified Prediction
            |
            v
     Streamlit Interface

```

## Model Architecture

### Intent Classification

```text
Customer Message
       |
       v
Text Preprocessing
       |
       v
TF-IDF Vectorization
       |
       v
Logistic Regression
       |
       v
Intent Prediction
       |
       +-----------------------------+
       |                             |
       v                             v
Supported Intent              Out-of-Scope

```
### Sentiment Classification

```text
Customer Message
       |
       v
Text Preprocessing
       |
       v
TF-IDF Vectorization
       |
       v
LinearSVC
       |
       v
Sentiment Prediction
       |
       +-----------------------------+
       |              |              |
       v              v              v
   Positive        Neutral        Negative

```

### PII Detection

```text
Customer Message
       |
       v
Text Preprocessing
       |
       v
TF-IDF Vectorization
       |
       v
Logistic Regression
       |
       v
PII Prediction
       |
       +----------------------+
       |                      |
       v                      v
    No PII               PII Present

```

## Project Structure

```text
trustlabel-ai/
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── customer_support_dataset_master.csv
│
├── docs/
│   ├── DevelopmentLog.md
│   ├── ProjectRoadmap.md
│   └── ResultsSummary.md
│
├── images/
│
├── models/
│   ├── intent_model.pkl
│   ├── intent_vectorizer.pkl
│   ├── sentiment_model.pkl
│   ├── sentiment_vectorizer.pkl
│   ├── pii_model.pkl
│   └── pii_vectorizer.pkl
│
└── src/
    ├── text_preprocessing.py
    ├── train_intent.py
    ├── train_sentiment.py
    ├── train_pii.py
    └── predictor.py
```

## Technologies Used

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- TF-IDF
- Logistic Regression
- LinearSVC

### Application

- Streamlit

### Development and Analysis

- Jupyter Notebook
- Matplotlib
