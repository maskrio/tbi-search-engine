import argparse
from bsbi import BSBIIndex
from compression import StandardPostings, VBEPostings, EliasGammaPostings

def main():
    parser = argparse.ArgumentParser(description="Demo pencarian dokumen")
    parser.add_argument("--compression", choices=["standard", "vbe", "elias-gamma"], default="vbe")
    parser.add_argument("--term-dict", choices=["idmap", "trie"], default="idmap")
    parser.add_argument("--scoring", choices=["tfidf", "bm25"], default="tfidf")
    parser.add_argument("--retrieval", choices=["full", "wand"], default="full")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    args = parser.parse_args()

    compression_map = {
        "standard": StandardPostings,
        "vbe": VBEPostings,
        "elias-gamma": EliasGammaPostings,
    }

    # sebelumnya sudah dilakukan indexing
    # BSBIIndex hanya sebagai abstraksi untuk index tersebut
    bsbi_instance = BSBIIndex(data_dir='collection', \
                              postings_encoding=compression_map[args.compression], \
                              output_dir='index',
                              term_dict_mode=args.term_dict)

    queries = ["alkylated with radioactive iodoacetate", \
               "psychodrama for disturbed children", \
               "lipid metabolism in toxemia and normal pregnancy"]

    for query in queries:
        print("Query  : ", query)
        print("Results:")
        if args.retrieval == "wand":
            results = bsbi_instance.retrieve_wand(query, k=args.k, scoring=args.scoring, k1=args.bm25_k1, b=args.bm25_b)
        elif args.scoring == "bm25":
            results = bsbi_instance.retrieve_bm25(query, k=args.k, k1=args.bm25_k1, b=args.bm25_b)
        else:
            results = bsbi_instance.retrieve_tfidf(query, k=args.k)

        for score, doc in results:
            print(f"{doc:30} {score:>.3f}")
        print()


if __name__ == '__main__':
    main()