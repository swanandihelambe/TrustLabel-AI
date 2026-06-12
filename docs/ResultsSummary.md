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
* Classes: Billing, Account, Technical, Security, Subscription

### Expanded Dataset

* Records: 500
* Same schema and intent labels

### Final Master Dataset

* Total Records: 625
* Balanced Intent Distribution:

  * Billing: 125
  * Account: 125
  * Technical: 125
  * Security: 125
  * Subscription: 125

---

## Intent Classification Results

| Model               |  Accuracy |
| ------------------- | --------: |
| Naive Bayes         |     70.4% |
| Logistic Regression | **72.8%** |
| Linear SVM          |     72.0% |

### Best Model

Logistic Regression (72.8%)

### Key Finding

Increasing dataset size from 125 to 625 samples significantly improved intent classification performance.

---

## Sentiment Classification Results

| Model               |  Accuracy |
| ------------------- | --------: |
| Naive Bayes         |     85.6% |
| Logistic Regression |     88.8% |
| Linear SVM          | **91.2%** |

### Best Model

Linear SVM (91.2%)

---

## PII Detection Results

### Initial Logistic Regression

* Accuracy: 87.2%
* Failed to identify PII messages due to class imbalance.

### Balanced Logistic Regression

* Accuracy: 98.4%
* PII Recall: 88%
* PII Precision: 100%

### Best Model

Balanced Logistic Regression

### Key Finding

Accuracy alone was misleading for imbalanced datasets. Applying class balancing significantly improved PII detection performance.

---

## Major Project Findings

1. Dataset expansion improved classification performance.
2. Logistic Regression performed best for intent classification.
3. Linear SVM performed best for sentiment classification.
4. Additional TF-IDF complexity did not improve results.
5. Class imbalance handling was critical for PII detection.

---

## Final Selected Models

| Task                     | Selected Model               |
| ------------------------ | ---------------------------- |
| Intent Classification    | Logistic Regression          |
| Sentiment Classification | Linear SVM                   |
| PII Detection            | Balanced Logistic Regression |

