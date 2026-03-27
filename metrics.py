import math


def rbp(ranking, p=0.8):
    """
    Menghitung Rank Biased Precision (RBP).

    Parameters
    ----------
    ranking: List[int]
        Vektor biner relevansi dokumen terurut berdasarkan ranking.
    p: float
        Parameter persistence RBP.

    Returns
    -------
    float
        Nilai RBP.
    """
    score = 0.0
    for i in range(1, len(ranking)):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking, k=10):
    """
    Menghitung Discounted Cumulative Gain (DCG) pada cutoff k.
    """
    score = 0.0
    for i, rel in enumerate(ranking[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg(ranking, k=10):
    """
    Menghitung Normalized Discounted Cumulative Gain (NDCG) pada cutoff k.
    """
    actual = dcg(ranking, k)
    ideal = dcg(sorted(ranking, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def ap(ranking, k=None):
    """
    Menghitung Average Precision (AP).

    Parameters
    ----------
    ranking: List[int]
        Vektor biner relevansi dokumen terurut.
    k: int atau None
        Cutoff evaluasi. Jika None maka gunakan seluruh ranking.

    Returns
    -------
    float
        Nilai AP.
    """
    if k is not None:
        ranking = ranking[:k]

    total_relevant = sum(ranking)
    if total_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_so_far = 0
    for i, rel in enumerate(ranking):
        if rel > 0:
            relevant_so_far += 1
            precision_sum += relevant_so_far / (i + 1)

    return precision_sum / total_relevant
