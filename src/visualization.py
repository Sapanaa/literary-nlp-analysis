import matplotlib.pyplot as plt
from pathlib import Path
from src.logger import get_logger

logger = get_logger(__name__)

IMAGES_DIR = Path("results/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def plot_top_words(top_words, show: bool = False):
    words, counts = zip(*top_words)

    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.xticks(rotation=45)
    plt.title("Top Frequent Words")
    plt.tight_layout()

    save_path = IMAGES_DIR / "top_words.png"
    plt.savefig(save_path, dpi=300)
    logger.info(f"Top words plot saved to {save_path}")

    


def plot_zipf(ranks, frequencies, show: bool = False):
    plt.figure(figsize=(6, 6))
    plt.loglog(ranks, frequencies)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf's Law (Word Frequency Distribution)")
    plt.tight_layout()

    save_path = IMAGES_DIR / "zipf_law.png"
    plt.savefig(save_path, dpi=300)
    logger.info(f"Zipf plot saved to {save_path}")

    
