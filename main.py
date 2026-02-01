from src.data_loader import download_moby_dick
from src.logger import get_logger

logger = get_logger(__name__)
RAW_TEXT_PATH = "data/raw/moby_dick.txt"


if __name__ == "__main__":
    logger.info("Pipeline Started")
    text = download_moby_dick(RAW_TEXT_PATH)
    print("Moby-Dick downloaded.")
    print(f"Number of characters: {len(text)}")
