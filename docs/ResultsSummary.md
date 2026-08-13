# TrustLabel AI - Results Summary

## Project Overview

TrustLabel AI is an NLP-powered customer support analytics system that classifies customer messages based on:

* Intent Classification
* Sentiment Classification
* PII Detection

The project uses TF-IDF feature engineering and Machine Learning models to analyze customer support messages.

---

## Dataset Information

### Original Dataset

* Records: 125
* Intent Classes:
  * Billing
  * Account
  * Technical
  * Security
  * Subscription

### Expanded Dataset

* Records: 625
* Same schema and original intent labels
* Dataset quality and duplicate checks completed

### Final Master Dataset

* Total Records: **750**
* Total Intent Classes: **6**

### Final Intent Distribution

* Account: 125
* Billing: 125
* Technical: 125
* Security: 125
* Subscription: 125
* Out-of-Scope: 125

The final dataset is balanced across all six intent classes.

---

## Intent Classification Results

### Intermediate Results - 625 Records

| Model | Accuracy |
|---|---:|
| Naive Bayes | 70.4% |
| Logistic Regression | **72.8%** |
| Linear SVM | 72.0% |

### Final Results - 750 Records

After introducing the **Out-of-Scope** class and retraining the intent classifier:

**Best Model: Logistic Regression**

**Final Accuracy: 80.7%**

### Final Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Account | 0.74 | 0.68 | 0.71 |
| Billing | 0.83 | 0.60 | 0.70 |
| Out-of-Scope | 0.85 | 0.88 | 0.86 |
| Security | 0.79 | 0.76 | 0.78 |
| Subscription | 0.96 | 1.00 | 0.98 |
| Technical | 0.70 | 0.92 | 0.79 |

### Key Finding

Increasing the dataset from 125 to 625 records improved the initial intent classification results. Adding a balanced **Out-of-Scope** class and retraining the final model further improved the measured accuracy from **72.8% to 80.7%** while allowing the system to explicitly identify messages outside the supported customer-support domain.

---

## Sentiment Classification Results

### Model Comparison

| Model | Accuracy |
|---|---:|
| Naive Bayes | 85.6% |
| Logistic Regression | 88.8% |
| Linear SVM / LinearSVC | **91.2%** |

### Best Model

**LinearSVC**

**Final Accuracy: 91.2%**

### Final Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Negative | 0.94 | 0.92 | 0.93 |
| Neutral | 0.88 | 0.95 | 0.91 |
| Positive | 0.93 | 0.84 | 0.88 |

---

## PII Detection Results

### Final Model

**Logistic Regression**

### Final Accuracy

**96.8%**

### Final Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| No PII | 0.96 | 1.00 | 0.98 |
| PII Present | 1.00 | 0.75 | 0.86 |

### Key Finding

Accuracy alone is not sufficient for evaluating PII detection because the test set contains substantially more non-PII examples.

The model achieved:

* PII Precision: **100%**
* PII Recall: **75%**
* PII F1-Score: **86%**

The 75% recall indicates that some PII-positive messages were missed by the classifier. Therefore, the model should not be treated as a security guarantee or used as the sole mechanism for protecting sensitive information.

---

## Major Project Findings

1. Increasing dataset size improved intent classification performance.
2. Logistic Regression performed best for the final intent classification task.
3. Adding an Out-of-Scope class improved domain handling and increased final intent accuracy to 80.7%.
4. LinearSVC performed best for sentiment classification.
5. Precision, recall, and F1-score provide more useful information than accuracy alone for PII detection.
6. PII-positive recall remains an important limitation of the current implementation.
7. A unified prediction pipeline successfully combines intent, sentiment, and PII predictions.
8. The final Streamlit application provides an interactive interface for testing the trained models.

---

## Final Selected Models

| Task | Selected Model | Features | Accuracy |
|---|---|---|---:|
| Intent Classification | Logistic Regression | TF-IDF | **80.7%** |
| Sentiment Classification | LinearSVC | TF-IDF | **91.2%** |
| PII Detection | Logistic Regression | TF-IDF | **96.8%** |

---

## Final System Status

The trained models, prediction pipeline, and Streamlit application have been completed and tested.

The project is ready for portfolio presentation.
