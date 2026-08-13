"""
TrustLabel AI

Module:
Sentiment Classification Training

Purpose:
Train and export the sentiment classification model.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from text_preprocessing import preprocess_text


# ============================================
# Project Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "customer_support_dataset_master.csv"

SENTIMENT_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.pkl"

SENTIMENT_VECTORIZER_PATH = PROJECT_ROOT / "models" / "sentiment_vectorizer.pkl"


def preprocess_messages(messages):
    """
    Apply text preprocessing to all customer support messages.
    """

    return messages.apply(preprocess_text)


def main():

    # Load Dataset
    dataset = pd.read_csv(DATASET_PATH)

    # Basic Information
    print("=" * 50)
    print("TrustLabel AI - Sentiment Classification")
    print("=" * 50)

    print(f"\nDataset Shape: {dataset.shape}")

    print("\nColumns:")
    print(dataset.columns.tolist())

    print("\nSentiment Classes:")
    print(sorted(dataset["sentiment"].unique()))

        # Extract Messages and Labels
    messages = dataset["message"]
    sentiment_labels = dataset["sentiment"]

    # Preprocess Messages
    clean_messages = preprocess_messages(messages)

    print("\nSample Preprocessed Messages:")
    print(clean_messages.head())

        # ============================================
    # TF-IDF Feature Engineering
    # ============================================

    sentiment_vectorizer = TfidfVectorizer(
        max_features=3000
    )

    sentiment_features = sentiment_vectorizer.fit_transform(
        clean_messages
    )

    print("\nTF-IDF Feature Matrix Shape:")
    print(sentiment_features.shape)

    print("\nNumber of Features:")
    print(len(sentiment_vectorizer.get_feature_names_out()))

        # ============================================
    # Train-Test Split
    # ============================================

    X_train, X_test, y_train, y_test = train_test_split(
        sentiment_features,
        sentiment_labels,
        test_size=0.20,
        random_state=42,
        stratify=sentiment_labels
    )

    print("\nTraining Data Shape:")
    print(X_train.shape)

    print("\nTesting Data Shape:")
    print(X_test.shape)

        # ============================================
    # Train Linear SVM Model
    # ============================================

    sentiment_model = LinearSVC(
        random_state=42
    )

    sentiment_model.fit(
        X_train,
        y_train
    )

    # Predict on Test Data
    sentiment_predictions = sentiment_model.predict(
        X_test
    )

    # Evaluate Model
    accuracy = accuracy_score(
        y_test,
        sentiment_predictions
    )

    print("\nSentiment Classification Accuracy:")
    print(f"{accuracy * 100:.1f}%")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            sentiment_predictions
        )
    )

        # ============================================
    # Save Model & Vectorizer
    # ============================================

    joblib.dump(
        sentiment_model,
        SENTIMENT_MODEL_PATH
    )

    joblib.dump(
        sentiment_vectorizer,
        SENTIMENT_VECTORIZER_PATH
    )

    # ============================================
    # Verify Saved Files
    # ============================================

    loaded_model = joblib.load(SENTIMENT_MODEL_PATH)
    loaded_vectorizer = joblib.load(SENTIMENT_VECTORIZER_PATH)

    print("\n" + "=" * 50)
    print("Verification")
    print("=" * 50)

    print("Classes:")
    print(loaded_model.classes_)

    print("\nModel Features:")
    print(loaded_model.n_features_in_)

    print("\nVectorizer Features:")
    print(len(loaded_vectorizer.get_feature_names_out()))

    print("\nModel Type:")
    print(type(loaded_model).__name__)

    print("\nVectorizer Type:")
    print(type(loaded_vectorizer).__name__)

    print("\nSentiment model and vectorizer verified successfully.")


if __name__ == "__main__":
    main()