from src.data_loader import download_moby_dick
from src.preprocessing import preprocess_text, save_clean_text
from src.logger import get_logger

logger = get_logger(__name__)

RAW_TEXT_PATH = "data/raw/moby_dick.txt"
CLEAN_TEXT_PATH = "data/processed/moby_dick_clean.txt"

if __name__ == "__main__":
    logger.info("Pipeline started")

    raw_text = download_moby_dick(RAW_TEXT_PATH)
    clean_tokens = preprocess_text(raw_text)
    save_clean_text(clean_tokens, CLEAN_TEXT_PATH)

    logger.info("Pipeline completed successfully")
