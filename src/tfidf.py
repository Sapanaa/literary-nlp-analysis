import re
from src.logger import get_logger
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pathlib import Path
logger = get_logger(__name__)


def split_into_chapters(text: str):
    """
    Split Moby-Dick into chapters using chapter headings.
    """
    logger.info("Splitting text into chapters")

    chapters = re.split(r"CHAPTER \d+\.", text)
    chapters = [ch.strip() for ch in chapters if len(ch.strip()) > 100]

    logger.info(f"Total chapters extracted: {len(chapters)}")
    return chapters



def build_tfidf(chapters, max_features=5000):
    """
    Build TF-IDF matrix from chapter texts.
    """
    logger.info("Building TF-IDF matrix")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )

    X = vectorizer.fit_transform(chapters)

    logger.info(f"TF-IDF shape: {X.shape}")
    return X, vectorizer


def get_top_tfidf_terms(X, vectorizer, top_k=20):
    """
    Get top TF-IDF terms across all documents.
    """
    feature_names = vectorizer.get_feature_names_out()
    mean_tfidf = np.asarray(X.mean(axis=0)).ravel()

    top_indices = mean_tfidf.argsort()[::-1][:top_k]
    return [(feature_names[i], mean_tfidf[i]) for i in top_indices]



def save_tfidf_terms(terms, save_path: str):
    """
    Save top TF-IDF terms to a text file.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for term, score in terms:
            f.write(f"{term}\t{score:.6f}\n")
