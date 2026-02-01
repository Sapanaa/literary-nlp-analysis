from collections import Counter
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)


def compute_text_statistics(tokens: list) -> dict:
    """
    Compute basic text statistics for a list of tokens.
    """
    logger.info("Computing text statistics")

    num_tokens = len(tokens)
    vocab_size = len(set(tokens))
    lexical_diversity = vocab_size / num_tokens if num_tokens > 0 else 0
    avg_word_length = np.mean([len(token) for token in tokens])

    stats = {
        "num_tokens": num_tokens,
        "vocab_size": vocab_size,
        "lexical_diversity": lexical_diversity,
        "avg_word_length": avg_word_length,
    }

    logger.info(f"Text statistics: {stats}")
    return stats


def get_top_words(tokens: list, n: int = 20):
    """
    Return the top-n most frequent words.
    """
    counter = Counter(tokens)
    return counter.most_common(n)

def zipf_distribution(tokens: list):
    """
    Compute word frequencies sorted by rank for Zipf's Law.
    """
    counter = Counter(tokens)
    frequencies = np.array(sorted(counter.values(), reverse=True))
    ranks = np.arange(1, len(frequencies) + 1)
    return ranks, frequencies
