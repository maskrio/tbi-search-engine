import heapq
import math


def retrieve_tfidf_taat(term_ids, merged_index, doc_id_map, k=10):
    """
    Melakukan retrieval TF-IDF dengan skema Term-at-a-Time (TaaT).

    Parameters
    ----------
    term_ids: List[int]
        Daftar term ID query yang valid (sudah ada di collection)
    merged_index: InvertedIndexReader
        Reader untuk main inverted index
    doc_id_map: IdMap
        Mapping docID -> nama dokumen
    k: int
        Banyak dokumen yang dikembalikan

    Returns
    -------
    List[Tuple[float, str]]
        Top-k dokumen terurut menurun berdasarkan score TF-IDF.
    """
    N = len(merged_index.doc_length)
    scores = {}

    for term in term_ids:
        if term not in merged_index.postings_dict:
            continue

        df = merged_index.postings_dict[term][1]
        idf = _tfidf_idf(N, df)
        postings, tf_list = merged_index.get_postings_list(term)

        for doc_id, tf in zip(postings, tf_list):
            if tf <= 0:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + _tfidf_term_score(idf, tf)

    return _top_k_docs(scores, doc_id_map, k)


def retrieve_bm25_taat(term_ids, merged_index, doc_id_map, k=10, k1=1.2, b=0.75):
    """
    Melakukan retrieval BM25 dengan skema Term-at-a-Time (TaaT).

    Parameters
    ----------
    term_ids: List[int]
        Daftar term ID query yang valid
    merged_index: InvertedIndexReader
        Reader untuk main inverted index
    doc_id_map: IdMap
        Mapping docID -> nama dokumen
    k: int
        Banyak dokumen yang dikembalikan
    k1: float
        Parameter saturasi TF untuk BM25
    b: float
        Parameter normalisasi panjang dokumen untuk BM25

    Returns
    -------
    List[Tuple[float, str]]
        Top-k dokumen terurut menurun berdasarkan score BM25.
    """
    N = len(merged_index.doc_length)
    avgdl = merged_index.avg_doc_length if merged_index.avg_doc_length > 0 else 1.0
    scores = {}

    for term in term_ids:
        if term not in merged_index.postings_dict:
            continue

        df = merged_index.postings_dict[term][1]
        idf = _bm25_idf(N, df)
        postings, tf_list = merged_index.get_postings_list(term)

        for doc_id, tf in zip(postings, tf_list):
            if tf <= 0:
                continue
            dl = merged_index.doc_length.get(doc_id, 0)
            score = _bm25_term_score(tf, dl, avgdl, idf, k1, b)
            scores[doc_id] = scores.get(doc_id, 0.0) + score

    return _top_k_docs(scores, doc_id_map, k)


def retrieve_wand(term_ids, merged_index, doc_id_map, k=10, scoring="bm25", k1=1.2, b=0.75):
    """
    Melakukan retrieval WAND top-k.

    WAND melakukan pruning kandidat menggunakan upper bound score per term,
    sehingga tidak semua dokumen harus dihitung skor penuhnya.

    Parameters
    ----------
    term_ids: List[int]
        Daftar term ID query yang valid
    merged_index: InvertedIndexReader
        Reader untuk main inverted index
    doc_id_map: IdMap
        Mapping docID -> nama dokumen
    k: int
        Banyak dokumen yang dikembalikan
    scoring: str
        Mode scoring: 'tfidf' atau 'bm25'
    k1: float
        Parameter BM25 (dipakai saat scoring='bm25')
    b: float
        Parameter BM25 (dipakai saat scoring='bm25')

    Returns
    -------
    List[Tuple[float, str]]
        Top-k dokumen terurut menurun berdasarkan score.
    """
    mode = scoring.lower()
    if mode not in ("tfidf", "bm25"):
        raise ValueError("scoring untuk WAND harus 'tfidf' atau 'bm25'")

    N = len(merged_index.doc_length)
    avgdl = merged_index.avg_doc_length if merged_index.avg_doc_length > 0 else 1.0
    min_dl = merged_index.min_doc_length if merged_index.min_doc_length > 0 else 1

    term_data = _build_wand_term_data(
        term_ids,
        merged_index,
        mode,
        N,
        avgdl,
        min_dl,
        k1,
        b,
    )

    if not term_data:
        return []

    heap = []
    threshold = float("-inf")

    while True:
        active = [td for td in term_data if td["ptr"] < len(td["postings"])]
        if not active:
            break

        active.sort(key=lambda td: td["postings"][td["ptr"]])

        score_ub = 0.0
        pivot_doc = None
        pivot_idx = -1
        for i, td in enumerate(active):
            score_ub += td["ub"]
            pivot_doc = td["postings"][td["ptr"]]
            if score_ub > threshold:
                pivot_idx = i
                break

        if pivot_idx == -1:
            break

        candidate_doc = active[0]["postings"][active[0]["ptr"]]

        if candidate_doc == pivot_doc:
            score = _score_candidate_doc(active, candidate_doc, mode, merged_index, avgdl, k1, b)

            if len(heap) < k:
                heapq.heappush(heap, (score, candidate_doc))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, candidate_doc))

            threshold = heap[0][0] if len(heap) == k else float("-inf")
        else:
            _advance_to_pivot(active[:pivot_idx], pivot_doc)

    results = sorted(heap, key=lambda x: x[0], reverse=True)
    return [(score, doc_id_map[doc_id]) for score, doc_id in results]


def _top_k_docs(scores, doc_id_map, k):
    docs = [(score, doc_id_map[doc_id]) for doc_id, score in scores.items()]
    return sorted(docs, key=lambda x: x[0], reverse=True)[:k]


def _build_wand_term_data(term_ids, merged_index, mode, N, avgdl, min_dl, k1, b):
    term_data = []
    for term in term_ids:
        if term not in merged_index.postings_dict:
            continue

        postings, tf_list = merged_index.get_postings_list(term)
        if not postings:
            continue

        df = merged_index.postings_dict[term][1]
        max_tf = merged_index.get_max_tf(term)
        idf = _tfidf_idf(N, df) if mode == "tfidf" else _bm25_idf(N, df)
        ub = _term_upper_bound(mode, idf, max_tf, min_dl, avgdl, k1, b)

        term_data.append({
            "postings": postings,
            "tf_list": tf_list,
            "ptr": 0,
            "idf": idf,
            "ub": ub,
        })
    return term_data


def _term_upper_bound(mode, idf, max_tf, min_dl, avgdl, k1, b):
    if max_tf <= 0:
        return 0.0
    if mode == "tfidf":
        return _tfidf_term_score(idf, max_tf)
    return _bm25_term_score(max_tf, min_dl, avgdl, idf, k1, b)


def _score_candidate_doc(active, candidate_doc, mode, merged_index, avgdl, k1, b):
    score = 0.0
    dl = merged_index.doc_length.get(candidate_doc, 0)

    for td in active:
        if td["postings"][td["ptr"]] != candidate_doc:
            continue

        tf = td["tf_list"][td["ptr"]]
        if mode == "tfidf":
            score += _tfidf_term_score(td["idf"], tf)
        else:
            score += _bm25_term_score(tf, dl, avgdl, td["idf"], k1, b)
        td["ptr"] += 1

    return score


def _advance_to_pivot(active_terms, pivot_doc):
    for td in active_terms:
        postings = td["postings"]
        while td["ptr"] < len(postings) and postings[td["ptr"]] < pivot_doc:
            td["ptr"] += 1


def _tfidf_idf(N, df):
    if df <= 0 or N <= 0:
        return 0.0
    return math.log(N / df)


def _bm25_idf(N, df):
    if df <= 0 or N <= 0:
        return 0.0
    return math.log(1 + ((N - df + 0.5) / (df + 0.5)))


def _tfidf_term_score(idf, tf):
    if tf <= 0:
        return 0.0
    return idf * (1 + math.log(tf))


def _bm25_term_score(tf, dl, avgdl, idf, k1, b):
    if tf <= 0:
        return 0.0
    if avgdl <= 0:
        avgdl = 1.0
    K = k1 * (1 - b + b * (dl / avgdl))
    return idf * ((tf * (k1 + 1)) / (tf + K))
