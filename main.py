from src.data_loader import download_moby_dick
from src.preprocessing import preprocess_text, save_clean_text, load_clean_text
from src.eda import compute_text_statistics, get_top_words, zipf_distribution
from src.visualization import plot_top_words, plot_zipf, plot_ngrams
from src.ngrams import get_ngrams, format_ngrams
from src.tfidf import split_into_chapters, build_tfidf, get_top_tfidf_terms, save_tfidf_terms
from src.topic_modelling import get_topic_terms, save_topics, run_nmf

from src.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)

RAW_TEXT_PATH = "data/raw/moby_dick.txt"
CLEAN_TEXT_PATH = "data/processed/moby_dick_clean.txt"

if __name__ == "__main__":
    logger.info("Pipeline started")
    # --- Load raw text (always needed for chapters) ---
    if Path(RAW_TEXT_PATH).exists():
        logger.info("Loading cached raw text")
        with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        logger.info("Downloading raw text")
        raw_text = download_moby_dick(RAW_TEXT_PATH)

    if Path(CLEAN_TEXT_PATH).exists():
        logger.info("Loading cached cleaned tokens")
        clean_tokens = load_clean_text(CLEAN_TEXT_PATH)
    else:
        logger.info("Preprocessing raw text")
        raw_text = download_moby_dick(RAW_TEXT_PATH)
        clean_tokens = preprocess_text(raw_text)
        save_clean_text(clean_tokens, CLEAN_TEXT_PATH)

    stats = compute_text_statistics(clean_tokens)
    top_words = get_top_words(clean_tokens, n=15)
    ranks, freqs = zipf_distribution(clean_tokens)

    plot_top_words(top_words)
    plot_zipf(ranks, freqs)

        # Bigram analysis
    bigrams = get_ngrams(clean_tokens, n=2, top_k=15)
    bigrams_fmt = format_ngrams(bigrams)
    plot_ngrams(
        bigrams_fmt,
        title="Top Bigrams",
        filename="top_bigrams_1.png"
    )

    # Trigram analysis
    trigrams = get_ngrams(clean_tokens, n=3, top_k=15)
    trigrams_fmt = format_ngrams(trigrams)
    plot_ngrams(
        trigrams_fmt,
        title="Top Trigrams",
        filename="top_trigrams_1.png"
    )

    # --- TF-IDF Feature Engineering ---
    chapters = split_into_chapters(raw_text)
    X_tfidf, vectorizer = build_tfidf(chapters)

    top_tfidf_terms = get_top_tfidf_terms(X_tfidf, vectorizer, top_k=20)
    TFIDF_TERMS_PATH = "results/tfidf/top_tfidf_terms.txt"

    save_tfidf_terms(top_tfidf_terms, TFIDF_TERMS_PATH)
    logger.info(f"Top TF-IDF terms saved to {TFIDF_TERMS_PATH}")

    # --- Topic Modeling ---
    nmf_model, W, H = run_nmf(X_tfidf, n_topics=6)

    topics = get_topic_terms(H, vectorizer, top_k=10)
    TOPICS_PATH = "results/topics/nmf_topics.txt"
    save_topics(topics, TOPICS_PATH)

    logger.info(f"Topics saved to {TOPICS_PATH}")

    logger.info("Pipeline completed successfully")
