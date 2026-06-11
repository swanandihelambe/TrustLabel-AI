# TrustLabel AI

A practical NLP and Machine Learning project inspired by a Data Labeling Analyst job simulation, focused on customer support ticket classification, sentiment analysis, and PII detection.

## Objective

Build a system that can:

* Classify customer support messages by intent
* Detect sentiment
* Identify PII (Personally Identifiable Information)
* Assist in label quality review

## Current Progress

### Phase 1: Dataset Creation ✅

* Created a labeled dataset of 125 customer support messages
* 5 intent categories:

  * Billing
  * Account
  * Technical
  * Security
  * Subscription
* Added Intent, Sentiment, and PII labels

---

### Phase 2: Exploratory Data Analysis (EDA) ✅

Performed:

* Dataset overview and quality checks
* Missing value analysis
* Intent distribution analysis
* Sentiment distribution analysis
* PII distribution analysis
* Intent vs Sentiment analysis

Key Findings:

* Dataset contains 125 labeled support messages
* Intent classes are balanced (25 examples each)
* Negative sentiment is the most common class
* Approximately 12% of messages contain PII
* Security-related messages show the highest negative sentiment

---

### Phase 3: NLP Preprocessing ✅

Implemented a text preprocessing pipeline:

* Converted text to lowercase
* Removed punctuation and special characters
* Tokenized messages into individual words
* Removed English stopwords
* Generated cleaned token lists for downstream machine learning tasks

Example:

Original:

"I was charged twice for my subscription."

Processed Tokens:

['charged', 'twice', 'subscription']

---

### Phase 4: TF-IDF Feature Engineering ✅

Implemented TF-IDF (Term Frequency–Inverse Document Frequency) to convert customer support messages into machine-learning-ready numerical features.

Completed:

* Converted processed tokens into text format
* Generated TF-IDF feature vectors
* Created a feature matrix of 125 messages and 244 unique features
* Identified the most influential words across the dataset

Key Findings:

* Generated 244 unique text features
* Frequently occurring support-related terms included:

  * account
  * update
  * subscription
  * email
  * password
  * refund

---

### Phase 5: Intent Classification ✅

Built a machine learning model to automatically classify customer support messages into intent categories.

Completed:

* Split dataset into training and testing sets (80/20)
* Trained a Multinomial Naive Bayes classifier
* Generated intent predictions on unseen test data
* Evaluated performance using multiple metrics

Results:

* Test Accuracy: 60%
* Classification Report generated
* Confusion Matrix generated

Key Insights:

* Billing and Subscription intents achieved the strongest performance
* Account and Security intents showed overlap due to similar vocabulary patterns
* The project now supports automatic intent prediction for customer support messages

---

### Phase 5A: Model Evaluation & Optimization 🔄

Current baseline models have been implemented using Multinomial Naive Bayes.

Current Results:

* Intent Classification Accuracy: 60%
* Sentiment Classification Accuracy: 56%

Observations:

* Baseline models successfully classify support messages and sentiments.
* Performance indicates vocabulary overlap between several classes, particularly Account vs Security and Neutral vs Positive sentiments.
* Additional model experimentation and optimization are planned to improve predictive performance.

Next Steps in Optimization:

* Compare multiple machine learning algorithms
* Evaluate Logistic Regression and other classifiers
* Select the best-performing model for deployment

---

## Tech Stack

* Python
* Pandas
* Matplotlib
* Scikit-learn
* NLTK
* Jupyter Notebook

---

## Next Steps

* Model Comparison & Optimization
* PII Detection System
* Label Review Assistant
* Streamlit Application Deployment

---

## Project Status

✅ Phase 1: Dataset Creation

✅ Phase 2: Exploratory Data Analysis (EDA)

✅ Phase 3: NLP Preprocessing

✅ Phase 4: TF-IDF Feature Engineering

✅ Phase 5: Intent Classification

🔄 Phase 5A: Model Evaluation & Optimization

⏳ Phase 6: PII Detection

⏳ Phase 7: Label Review Assistant

⏳ Phase 8: Streamlit Application
