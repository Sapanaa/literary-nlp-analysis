from collections import Counter
from nltk.util import ngrams
from src.logger import get_logger

logger = get_logger(__name__)


def get_ngrams(tokens: list, n: int = 2, top_k: int = 20):
    """
    Compute top-k most frequent n-grams.
    """
    logger.info(f"Computing {n}-grams")

    ngram_list = ngrams(tokens, n)
    ngram_counts = Counter(ngram_list)

    top_ngrams = ngram_counts.most_common(top_k)

    logger.info(f"Top {n}-grams computed")
    return top_ngrams


def format_ngrams(ngrams_list):
    """
    Convert n-gram tuples into readable strings.
    """
    return [(" ".join(ngram), count) for ngram, count in ngrams_list]
