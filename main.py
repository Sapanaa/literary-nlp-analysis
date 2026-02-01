from src.data_loader import download_moby_dick
from src.preprocessing import preprocess_text, save_clean_text
from src.eda import compute_text_statistics, get_top_words, zipf_distribution
from src.visualization import plot_top_words, plot_zipf
from src.logger import get_logger

logger = get_logger(__name__)

RAW_TEXT_PATH = "data/raw/moby_dick.txt"
CLEAN_TEXT_PATH = "data/processed/moby_dick_clean.txt"

if __name__ == "__main__":
    logger.info("Pipeline started")

    raw_text = download_moby_dick(RAW_TEXT_PATH)
    clean_tokens = preprocess_text(raw_text)
    save_clean_text(clean_tokens, CLEAN_TEXT_PATH)

    stats = compute_text_statistics(clean_tokens)
    top_words = get_top_words(clean_tokens, n=15)
    ranks, freqs = zipf_distribution(clean_tokens)

    plot_top_words(top_words)
    plot_zipf(ranks, freqs)

    logger.info("Pipeline completed successfully")
