import argparse
import os
from bsbi import BSBIIndex
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from metrics import rbp, dcg, ndcg, ap
from vector_index import LSIFaissIndex

######## >>>>> sebuah IR metric: RBP p = 0.8


######## >>>>> memuat qrels


def load_qrels(qrel_file="qrels.txt", max_q_id=30, max_doc_id=1033):
    """memuat query relevance judgment (qrels)
    dalam format dictionary of dictionary
    qrels[query id][document id]

    dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
    relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
    Doc 10 tidak relevan dengan Q3.

    """
    qrels = {
        "Q" + str(i): {i: 0 for i in range(1, max_doc_id + 1)}
        for i in range(1, max_q_id + 1)
    }
    with open(qrel_file) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels


######## >>>>> EVALUASI !


def eval(
    qrels,
    query_file="queries.txt",
    k=1000,
    compression="vbe",
    term_dict_mode="idmap",
    scoring="tfidf",
    retrieval="full",
    bm25_k1=1.2,
    bm25_b=0.75,
    output_dir="index",
    vector_dir="index",
):
    """
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
    """
    compression_map = {
        "standard": StandardPostings,
        "vbe": VBEPostings,
        "elias-gamma": EliasGammaPostings,
    }

    BSBI_instance = BSBIIndex(
        data_dir="collection",
        postings_encoding=compression_map[compression],
        output_dir=output_dir,
        term_dict_mode=term_dict_mode,
    )

    vector_instance = None
    if retrieval == "faiss":
        vector_instance = LSIFaissIndex(data_dir="collection", output_dir=vector_dir)
        if not vector_instance.has_artifacts():
            raise FileNotFoundError(
                "Vector index belum tersedia. Jalankan: python vector_build.py"
            )
        vector_instance.load()

    def retrieve(query_text):
        if retrieval == "faiss":
            return vector_instance.query_faiss(query_text, k=k)
        if retrieval == "wand":
            return BSBI_instance.retrieve_wand(
                query_text, k=k, scoring=scoring, k1=bm25_k1, b=bm25_b
            )
        if scoring == "bm25":
            return BSBI_instance.retrieve_bm25(query_text, k=k, k1=bm25_k1, b=bm25_b)
        return BSBI_instance.retrieve_tfidf(query_text, k=k)

    with open(query_file) as file:
        rbp_scores = []
        dcg_scores = []
        ndcg_scores = []
        ap_scores = []
        for qline in file:
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
            # yang tertera di qrels
            ranking = []
            for score, doc in retrieve(query):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                ranking.append(qrels[qid][did])

            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking, k=min(10, len(ranking))))
            ndcg_scores.append(ndcg(ranking, k=min(10, len(ranking))))
            ap_scores.append(ap(ranking))

    if retrieval == "faiss":
        method_label = "LSI+FAISS"
    else:
        method_label = f"{scoring.upper()} + {retrieval.upper()}"

    print(f"Hasil evaluasi ({method_label}) terhadap 30 queries")
    print("RBP score  =", sum(rbp_scores) / len(rbp_scores))
    print("DCG score  =", sum(dcg_scores) / len(dcg_scores))
    print("NDCG score =", sum(ndcg_scores) / len(ndcg_scores))
    print("AP score   =", sum(ap_scores) / len(ap_scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi retrieval")
    parser.add_argument("--query-file", default="queries.txt")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument(
        "--compression", choices=["standard", "vbe", "elias-gamma"], default="vbe"
    )
    parser.add_argument("--term-dict", choices=["idmap", "trie"], default="idmap")
    parser.add_argument("--scoring", choices=["tfidf", "bm25"], default="tfidf")
    parser.add_argument("--retrieval", choices=["full", "wand", "faiss"], default="full")
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--output-dir", default="index")
    parser.add_argument("--vector-dir", default="index")
    args = parser.parse_args()

    qrels = load_qrels()

    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"

    eval(
        qrels,
        query_file=args.query_file,
        k=args.k,
        compression=args.compression,
        term_dict_mode=args.term_dict,
        scoring=args.scoring,
        retrieval=args.retrieval,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        output_dir=args.output_dir,
        vector_dir=args.vector_dir,
    )
