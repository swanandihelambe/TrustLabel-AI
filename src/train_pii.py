"""
TrustLabel AI

Module:
PII Detection Training

Purpose:
Train and export the PII detection model.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC

from text_preprocessing import preprocess_text


# ============================================
# Project Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "customer_support_dataset_master.csv"

PII_MODEL_PATH = PROJECT_ROOT / "models" / "pii_model.pkl"

PII_VECTORIZER_PATH = PROJECT_ROOT / "models" / "pii_vectorizer.pkl"


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
    print("TrustLabel AI - PII Detection")
    print("=" * 50)

    print(f"\nDataset Shape: {dataset.shape}")

    print("\nColumns:")
    print(dataset.columns.tolist())

    print("\nPII Classes:")
    print(sorted(dataset["pii"].unique()))

        # Extract Messages and Labels
    messages = dataset["message"]
    pii_labels = dataset["pii"]

    # Preprocess Messages
    clean_messages = preprocess_messages(messages)

    print("\nSample Preprocessed Messages:")
    print(clean_messages.head())

        # ============================================
    # TF-IDF Feature Engineering
    # ============================================

    pii_vectorizer = TfidfVectorizer(
        max_features=3000
    )

    pii_features = pii_vectorizer.fit_transform(
        clean_messages
    )

    print("\nTF-IDF Feature Matrix Shape:")
    print(pii_features.shape)

    print("\nNumber of Features:")
    print(len(pii_vectorizer.get_feature_names_out()))

        # ============================================
    # Train-Test Split
    # ============================================

    X_train, X_test, y_train, y_test = train_test_split(
        pii_features,
        pii_labels,
        test_size=0.20,
        random_state=42,
        stratify=pii_labels
    )

    print("\nTraining Data Shape:")
    print(X_train.shape)

    print("\nTesting Data Shape:")
    print(X_test.shape)

        # ============================================
    # Train Linear SVM Model
    # ============================================

    pii_model = LinearSVC(
        random_state=42
    )

    pii_model.fit(
        X_train,
        y_train
    )

    # Predict on Test Data
    pii_predictions = pii_model.predict(
        X_test
    )

    # Evaluate Model
    accuracy = accuracy_score(
        y_test,
        pii_predictions
    )

    print("\nPII Detection Accuracy:")
    print(f"{accuracy * 100:.1f}%")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            pii_predictions
        )
    )

        # ============================================
    # Save Model & Vectorizer
    # ============================================

    joblib.dump(
        pii_model,
        PII_MODEL_PATH
    )

    joblib.dump(
        pii_vectorizer,
        PII_VECTORIZER_PATH
    )

    # ============================================
    # Verify Saved Files
    # ============================================

    loaded_model = joblib.load(PII_MODEL_PATH)
    loaded_vectorizer = joblib.load(PII_VECTORIZER_PATH)

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

    print("\nPII model and vectorizer verified successfully.")


if __name__ == "__main__":
    main()