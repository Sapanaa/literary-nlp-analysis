import numpy as np
from sklearn.decomposition import NMF
from src.logger import get_logger

logger = get_logger(__name__)


def run_nmf(X, n_topics=6, random_state=42):
    """
    Fit an NMF topic model on a TF-IDF matrix.
    """
    logger.info(f"Running NMF with {n_topics} topics")

    nmf = NMF(
        n_components=n_topics,
        random_state=random_state,
        init="nndsvd",
        max_iter=500
    )

    W = nmf.fit_transform(X)  # document-topic matrix
    H = nmf.components_       # topic-term matrix

    logger.info("NMF training completed")
    return nmf, W, H

def get_topic_terms(H, vectorizer, top_k=10):
    """
    Get top terms for each topic.
    """
    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic_idx, topic_weights in enumerate(H):
        top_indices = topic_weights.argsort()[::-1][:top_k]
        top_terms = [(feature_names[i], topic_weights[i]) for i in top_indices]
        topics.append((topic_idx, top_terms))

    return topics

from pathlib import Path

def save_topics(topics, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for topic_id, terms in topics:
            f.write(f"Topic {topic_id}\n")
            for term, weight in terms:
                f.write(f"  {term}\t{weight:.4f}\n")
            f.write("\n")
