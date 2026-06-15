# TrustLabel AI

TrustLabel AI is an NLP-based customer support analytics system that automatically analyzes customer messages and predicts:

* Intent (Billing, Account, Technical, Security, Subscription)
* Sentiment (Positive, Neutral, Negative)
* PII Presence (Yes/No)

The project uses TF-IDF feature engineering and Machine Learning models to automate customer support ticket analysis.

---

## Key Results

| Task                     | Best Model                   |         Result |
| ------------------------ | ---------------------------- | -------------: |
| Intent Classification    | Logistic Regression          | 72.8% Accuracy |
| Sentiment Classification | Linear SVM                   | 91.2% Accuracy |
| PII Detection            | Balanced Logistic Regression | 98.4% Accuracy |

Dataset Size: **625 labeled customer support messages**

---

## Features

* Intent Classification
* Sentiment Analysis
* PII Detection
* TF-IDF Feature Engineering
* Multi-model Evaluation and Comparison
* Exported Models for Deployment

---

## Machine Learning Pipeline

1. Data Collection and Labeling
2. Exploratory Data Analysis (EDA)
3. NLP Preprocessing
4. TF-IDF Vectorization
5. Model Training
6. Model Evaluation
7. Model Export

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
└── requirements.txt
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

## Key Learnings

* Increasing dataset size from 125 to 625 records significantly improved performance.
* Logistic Regression performed best for intent classification.
* Linear SVM achieved the highest sentiment classification accuracy.
* Class imbalance handling was critical for successful PII detection.

---

## Future Enhancements

* Unified prediction pipeline
* Streamlit web application
* Prediction confidence scores
* Expanded customer support dataset

---
