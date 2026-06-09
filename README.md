# TrustLabel AI

An NLP-based project inspired by a Data Labeling job simulation.

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

## Tech Stack

* Python
* Pandas
* Matplotlib
* NLTK
* Jupyter Notebook

## Next Steps

* TF-IDF Feature Engineering
* Intent Classification Model
* Sentiment Classification Model
* PII Detection System
* Label Review Assistant
