"""
TrustLabel AI

Module:
Intent Classification Training

Purpose:
Train and export the intent classification model.
"""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from text_preprocessing import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ============================================
# Project Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "customer_support_dataset_master.csv"

INTENT_MODEL_PATH = PROJECT_ROOT / "models" / "intent_model.pkl"

INTENT_VECTORIZER_PATH = PROJECT_ROOT / "models" / "intent_vectorizer.pkl"

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
    print("TrustLabel AI - Intent Classification")
    print("=" * 50)

    print(f"\nDataset Shape: {dataset.shape}")

    print("\nColumns:")
    print(dataset.columns.tolist())

    print("\nIntent Classes:")
    print(sorted(dataset["intent"].unique()))

    # Extract Messages and Labels
    messages = dataset["message"]
    intent_labels = dataset["intent"]

    # Preprocess Messages
    clean_messages = preprocess_messages(messages)

    # ============================================
    # TF-IDF Feature Engineering
    # ============================================

    intent_vectorizer = TfidfVectorizer(
        max_features=3000
    )

    intent_features = intent_vectorizer.fit_transform(clean_messages)

    # ============================================
    # Train-Test Split
    # ============================================

    X_train, X_test, y_train, y_test = train_test_split(
        intent_features,
        intent_labels,
        test_size=0.20,
        random_state=42,
        stratify=intent_labels
    )

    # ============================================
    # Train Logistic Regression Model
    # ============================================

    intent_model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    intent_model.fit(X_train, y_train)

    # Predict on Test Data
    intent_predictions = intent_model.predict(X_test)

    # Evaluate Model
    accuracy = accuracy_score(
        y_test,
        intent_predictions
    )

    
    print("\nSample Preprocessed Messages:")
    print(clean_messages.head())

    print("\nTF-IDF Feature Matrix Shape:")
    print(intent_features.shape)

    print("\nNumber of Features:")
    print(len(intent_vectorizer.get_feature_names_out()))

    print("\nTraining Data Shape:")
    print(X_train.shape)

    print("\nTesting Data Shape:")
    print(X_test.shape)

    print("\nIntent Classification Accuracy:")
    print(f"{accuracy * 100:.1f}%")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            intent_predictions
        )
    )

    # ============================================
    # Save Model & Vectorizer
    # ============================================

    joblib.dump(
    intent_vectorizer,
    INTENT_VECTORIZER_PATH
)

# ============================================
# Verify Saved Files
# ============================================

loaded_model = joblib.load(INTENT_MODEL_PATH)
loaded_vectorizer = joblib.load(INTENT_VECTORIZER_PATH)

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

print("\nIntent model and vectorizer verified successfully.")

if __name__ == "__main__":
    main()