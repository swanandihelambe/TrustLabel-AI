# ============================================
# Text preprocessing utilities for TrustLabel AI.
# ============================================

import re

def preprocess_text(text):
    """
    Clean customer support messages before vectorization.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text