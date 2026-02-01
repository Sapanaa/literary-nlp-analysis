import spacy
from nltk.corpus import stopwords
from src.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Load stopwords
STOP_WORDS = set(stopwords.words("english"))


def preprocess_text(text: str, chunk_size: int = 500_000) -> list:
    """
    Clean and preprocess raw text in chunks to avoid memory issues.
    """
    logger.info("Starting text preprocessing")

    tokens = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        doc = nlp(chunk)

        tokens.extend(
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
            and token.text.lower() not in STOP_WORDS
        )

        logger.info(
            f"Processed chunk {i // chunk_size + 1}"
        )

    logger.info(f"Preprocessing complete. Total tokens: {len(tokens)}")
    return tokens


def save_clean_text(tokens: list, save_path: str):
    logger.info(f"Saving cleaned text to {save_path}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(" ".join(tokens))