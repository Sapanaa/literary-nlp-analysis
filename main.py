from src.data_loader import download_moby_dick
from src.preprocessing import preprocess_text, save_clean_text, load_clean_text
from src.eda import compute_text_statistics, get_top_words, zipf_distribution
from src.visualization import plot_top_words, plot_zipf, plot_ngrams
from src.ngrams import get_ngrams, format_ngrams
from src.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)

RAW_TEXT_PATH = "data/raw/moby_dick.txt"
CLEAN_TEXT_PATH = "data/processed/moby_dick_clean.txt"

if __name__ == "__main__":
    logger.info("Pipeline started")

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
    logger.info("Pipeline completed successfully")
