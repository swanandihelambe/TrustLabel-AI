#Load models & predict

"""
TrustLabel AI

Module:
Unified Prediction Pipeline

Purpose:
Load trained models and generate intent, sentiment,
and PII predictions for customer support messages.
"""

from pathlib import Path

import joblib

from src.text_preprocessing import preprocess_text


# ============================================
# Project Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INTENT_MODEL_PATH = PROJECT_ROOT / "models" / "intent_model.pkl"
INTENT_VECTORIZER_PATH = PROJECT_ROOT / "models" / "intent_vectorizer.pkl"

SENTIMENT_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.pkl"
SENTIMENT_VECTORIZER_PATH = PROJECT_ROOT / "models" / "sentiment_vectorizer.pkl"

PII_MODEL_PATH = PROJECT_ROOT / "models" / "pii_model.pkl"
PII_VECTORIZER_PATH = PROJECT_ROOT / "models" / "pii_vectorizer.pkl"


# ============================================
# Load Trained Models & Vectorizers
# ============================================

intent_model = joblib.load(INTENT_MODEL_PATH)
intent_vectorizer = joblib.load(INTENT_VECTORIZER_PATH)

sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
sentiment_vectorizer = joblib.load(SENTIMENT_VECTORIZER_PATH)

pii_model = joblib.load(PII_MODEL_PATH)
pii_vectorizer = joblib.load(PII_VECTORIZER_PATH)


# ============================================
# Validate Model-Vectorizer Compatibility
# ============================================

assert (
    intent_model.n_features_in_
    == len(intent_vectorizer.get_feature_names_out())
), "Intent model/vectorizer feature mismatch."

assert (
    sentiment_model.n_features_in_
    == len(sentiment_vectorizer.get_feature_names_out())
), "Sentiment model/vectorizer feature mismatch."

assert (
    pii_model.n_features_in_
    == len(pii_vectorizer.get_feature_names_out())
), "PII model/vectorizer feature mismatch."


print("All models and vectorizers loaded successfully.")
print("Feature compatibility verified.")

# ============================================
# Unified Prediction Function
# ============================================

def predict_message(message):
    """
    Predict intent, sentiment, and PII for a customer support message.
    """

    # Preprocess input message
    clean_message = preprocess_text(message)

    # ========================================
    # Intent Prediction
    # ========================================

    intent_features = intent_vectorizer.transform(
        [clean_message]
    )

    intent_prediction = intent_model.predict(
        intent_features
    )[0]

    # Intent confidence
    intent_probabilities = intent_model.predict_proba(
        intent_features
    )

    intent_confidence = intent_probabilities.max()

    # ========================================
    # Sentiment Prediction
    # ========================================

    sentiment_features = sentiment_vectorizer.transform(
        [clean_message]
    )

    sentiment_prediction = sentiment_model.predict(
        sentiment_features
    )[0]

    # ========================================
    # PII Prediction
    # ========================================

    pii_features = pii_vectorizer.transform(
        [clean_message]
    )

    pii_prediction = pii_model.predict(
        pii_features
    )[0]

    # ========================================
    # Return Unified Result
    # ========================================

    return {
        "intent": intent_prediction,
        "intent_confidence": intent_confidence,
        "sentiment": sentiment_prediction,
        "pii": pii_prediction
    }

# ============================================
# Test Prediction
# ============================================

'''if __name__ == "__main__":

    test_messages = [
    "I cannot log into my account.",
    "The app keeps crashing whenever I try to open it.",
    "Someone used my card without my permission.",
    "My email address is example@gmail.com and I need to change my password."
]

for message in test_messages:

    result = predict_message(message)

    print("\n" + "-" * 50)
    print(f"Message: {message}")
    print(f"Prediction: {result}")

    print("\n" + "=" * 50)
    print("TrustLabel AI - Prediction Test")
    print("=" * 50)

   print(f"\nMessage:")
    print(sample_message)

    print("\nPrediction:")
    print(result)'''